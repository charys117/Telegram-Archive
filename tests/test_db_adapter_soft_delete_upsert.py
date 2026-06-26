import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from src.db.adapter import DatabaseAdapter
from src.db.base import DatabaseManager
from src.db.models import Message, MessageVersion


@pytest.fixture
async def sqlite_adapter(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path / 'telegram_archive.db'}")
    await manager.init()
    try:
        yield DatabaseAdapter(manager)
    finally:
        await manager.close()


async def _get_message(adapter: DatabaseAdapter, message_id: int, chat_id: int) -> Message:
    async with adapter.db_manager.async_session_factory() as session:
        message = await session.get(Message, (message_id, chat_id))
        assert message is not None
        return message


async def _get_versions(adapter: DatabaseAdapter, message_id: int, chat_id: int) -> list[MessageVersion]:
    async with adapter.db_manager.async_session_factory() as session:
        result = await session.execute(
            select(MessageVersion)
            .where(MessageVersion.message_id == message_id, MessageVersion.chat_id == chat_id)
            .order_by(MessageVersion.id.asc())
        )
        return list(result.scalars())


@pytest.mark.asyncio
async def test_insert_message_upsert_preserves_soft_delete_marker(sqlite_adapter):
    deleted_at = datetime(2026, 6, 25, 10, 30)

    await sqlite_adapter.insert_message(
        {
            "id": 1,
            "chat_id": 100,
            "date": datetime(2026, 6, 25, 10, 0),
            "text": "original",
        }
    )
    await sqlite_adapter.mark_message_deleted(100, 1, deleted_at)

    await sqlite_adapter.insert_message(
        {
            "id": 1,
            "chat_id": 100,
            "date": datetime(2026, 6, 25, 10, 0),
            "text": "reprocessed",
        }
    )

    message = await _get_message(sqlite_adapter, 1, 100)
    assert message.text == "original"
    assert message.is_deleted == 1
    assert message.deleted_at == deleted_at


@pytest.mark.asyncio
async def test_insert_messages_batch_upsert_preserves_soft_delete_marker(sqlite_adapter):
    deleted_at = datetime(2026, 6, 25, 11, 30)

    await sqlite_adapter.insert_messages_batch(
        [
            {
                "id": 2,
                "chat_id": 100,
                "date": datetime(2026, 6, 25, 11, 0),
                "text": "original",
            }
        ]
    )
    await sqlite_adapter.mark_message_deleted(100, 2, deleted_at)

    await sqlite_adapter.insert_messages_batch(
        [
            {
                "id": 2,
                "chat_id": 100,
                "date": datetime(2026, 6, 25, 11, 0),
                "text": "reprocessed",
            }
        ]
    )

    message = await _get_message(sqlite_adapter, 2, 100)
    assert message.text == "original"
    assert message.is_deleted == 1
    assert message.deleted_at == deleted_at


@pytest.mark.asyncio
async def test_fresh_insert_not_marked_deleted(sqlite_adapter):
    """A brand-new message inserts as not-deleted with a null deleted_at."""
    await sqlite_adapter.insert_message(
        {"id": 3, "chat_id": 100, "date": datetime(2026, 6, 25, 12, 0), "text": "hello"}
    )

    message = await _get_message(sqlite_adapter, 3, 100)
    assert message.is_deleted == 0
    assert message.deleted_at is None


@pytest.mark.asyncio
async def test_upsert_with_is_deleted_and_timestamp_sets_marker(sqlite_adapter):
    """An upsert whose payload carries is_deleted + deleted_at sets both on conflict."""
    deleted_at = datetime(2026, 6, 25, 13, 30)

    await sqlite_adapter.insert_message(
        {"id": 4, "chat_id": 100, "date": datetime(2026, 6, 25, 13, 0), "text": "original"}
    )
    await sqlite_adapter.insert_message(
        {
            "id": 4,
            "chat_id": 100,
            "date": datetime(2026, 6, 25, 13, 0),
            "text": "now deleted",
            "is_deleted": 1,
            "deleted_at": deleted_at,
        }
    )

    message = await _get_message(sqlite_adapter, 4, 100)
    assert message.is_deleted == 1
    assert message.deleted_at == deleted_at


@pytest.mark.asyncio
async def test_upsert_with_is_deleted_no_timestamp_preserves_existing(sqlite_adapter):
    """An upsert with is_deleted=1 but no deleted_at keeps the existing timestamp."""
    first_deleted_at = datetime(2026, 6, 25, 14, 30)

    await sqlite_adapter.insert_message(
        {"id": 5, "chat_id": 100, "date": datetime(2026, 6, 25, 14, 0), "text": "original"}
    )
    await sqlite_adapter.mark_message_deleted(100, 5, first_deleted_at)
    await sqlite_adapter.insert_message(
        {
            "id": 5,
            "chat_id": 100,
            "date": datetime(2026, 6, 25, 14, 0),
            "text": "reprocessed",
            "is_deleted": 1,
        }
    )

    message = await _get_message(sqlite_adapter, 5, 100)
    assert message.is_deleted == 1
    assert message.deleted_at == first_deleted_at


@pytest.mark.asyncio
async def test_mark_message_deleted_twice_keeps_first_timestamp(sqlite_adapter):
    """Re-marking a soft-deleted message preserves the original deletion time (coalesce)."""
    first = datetime(2026, 6, 25, 15, 0)
    second = datetime(2026, 6, 25, 16, 0)

    await sqlite_adapter.insert_message(
        {"id": 6, "chat_id": 100, "date": datetime(2026, 6, 25, 14, 0), "text": "original"}
    )
    await sqlite_adapter.mark_message_deleted(100, 6, first)
    await sqlite_adapter.mark_message_deleted(100, 6, second)

    message = await _get_message(sqlite_adapter, 6, 100)
    assert message.is_deleted == 1
    assert message.deleted_at == first


@pytest.mark.asyncio
async def test_mark_message_deleted_defaults_timestamp_when_none(sqlite_adapter):
    """Omitting deleted_at falls back to a server-generated timestamp."""
    await sqlite_adapter.insert_message(
        {"id": 7, "chat_id": 100, "date": datetime(2026, 6, 25, 14, 0), "text": "original"}
    )
    await sqlite_adapter.mark_message_deleted(100, 7)

    message = await _get_message(sqlite_adapter, 7, 100)
    assert message.is_deleted == 1
    assert message.deleted_at is not None


@pytest.mark.asyncio
async def test_get_messages_sync_data_excludes_soft_deleted(sqlite_adapter):
    """Soft-deleted rows are excluded from the sync set so they aren't re-checked."""
    await sqlite_adapter.insert_message(
        {"id": 10, "chat_id": 200, "date": datetime(2026, 6, 25, 17, 0), "text": "live"}
    )
    await sqlite_adapter.insert_message(
        {"id": 11, "chat_id": 200, "date": datetime(2026, 6, 25, 17, 1), "text": "to delete"}
    )
    await sqlite_adapter.mark_message_deleted(200, 11)

    sync_data = await sqlite_adapter.get_messages_sync_data(200)
    assert set(sync_data.keys()) == {10}


@pytest.mark.asyncio
async def test_update_message_text_records_previous_version(sqlite_adapter):
    await sqlite_adapter.insert_message(
        {"id": 20, "chat_id": 300, "date": datetime(2026, 6, 26, 9, 0), "text": "original"}
    )

    edit_date = datetime(2026, 6, 26, 9, 5)
    await sqlite_adapter.update_message_text(300, 20, "edited", edit_date, source="listener_edit")

    message = await _get_message(sqlite_adapter, 20, 300)
    versions = await _get_versions(sqlite_adapter, 20, 300)
    assert message.text == "edited"
    assert message.edit_date == edit_date
    assert len(versions) == 1
    assert versions[0].text == "original"
    assert versions[0].date == datetime(2026, 6, 26, 9, 0)


@pytest.mark.asyncio
async def test_update_message_text_is_idempotent(sqlite_adapter):
    edit_date = datetime(2026, 6, 26, 10, 5)
    await sqlite_adapter.insert_message(
        {"id": 21, "chat_id": 300, "date": datetime(2026, 6, 26, 10, 0), "text": "original"}
    )

    await sqlite_adapter.update_message_text(300, 21, "edited", edit_date, source="listener_edit")
    await sqlite_adapter.update_message_text(300, 21, "edited", edit_date, source="sync_edit")

    versions = await _get_versions(sqlite_adapter, 21, 300)
    assert len(versions) == 1
    assert versions[0].text == "original"


@pytest.mark.asyncio
async def test_update_message_text_same_text_updates_edit_date_without_version(sqlite_adapter):
    await sqlite_adapter.insert_message(
        {"id": 28, "chat_id": 300, "date": datetime(2026, 6, 26, 10, 0), "text": "original"}
    )

    reaction_edit_date = datetime(2026, 6, 26, 10, 15)
    await sqlite_adapter.update_message_text(300, 28, "original", reaction_edit_date, source="sync_edit")

    message = await _get_message(sqlite_adapter, 28, 300)
    versions = await _get_versions(sqlite_adapter, 28, 300)
    assert message.text == "original"
    assert message.edit_date == reaction_edit_date
    assert versions == []


@pytest.mark.asyncio
async def test_update_message_text_older_edit_date_does_not_roll_back(sqlite_adapter):
    current_edit_date = datetime(2026, 6, 26, 10, 30)
    old_edit_date = datetime(2026, 6, 26, 10, 10)
    await sqlite_adapter.insert_message(
        {
            "id": 27,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 10, 0),
            "text": "current",
            "edit_date": current_edit_date,
        }
    )

    await sqlite_adapter.update_message_text(300, 27, "older", old_edit_date, source="listener_edit")

    message = await _get_message(sqlite_adapter, 27, 300)
    versions = await _get_versions(sqlite_adapter, 27, 300)
    assert message.text == "current"
    assert message.edit_date == current_edit_date
    assert versions == []


@pytest.mark.asyncio
async def test_text_only_edit_records_previous_version(sqlite_adapter):
    await sqlite_adapter.insert_message(
        {"id": 22, "chat_id": 300, "date": datetime(2026, 6, 26, 11, 0), "text": "caption"}
    )

    await sqlite_adapter.update_message_text(300, 22, "caption edited", None, source="listener_edit")

    message = await _get_message(sqlite_adapter, 22, 300)
    versions = await _get_versions(sqlite_adapter, 22, 300)
    assert message.text == "caption edited"
    assert message.edit_date is None
    assert len(versions) == 1
    assert versions[0].text == "caption"
    assert versions[0].date == datetime(2026, 6, 26, 11, 0)


@pytest.mark.asyncio
async def test_upsert_with_newer_edit_date_records_previous_version(sqlite_adapter):
    await sqlite_adapter.insert_message(
        {"id": 23, "chat_id": 300, "date": datetime(2026, 6, 26, 12, 0), "text": "original"}
    )
    edit_date = datetime(2026, 6, 26, 12, 30)

    await sqlite_adapter.insert_message(
        {
            "id": 23,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "edited via backup",
            "edit_date": edit_date,
        },
        source="backup_upsert",
    )

    message = await _get_message(sqlite_adapter, 23, 300)
    versions = await _get_versions(sqlite_adapter, 23, 300)
    assert message.text == "edited via backup"
    assert message.edit_date == edit_date
    assert len(versions) == 1
    assert versions[0].text == "original"
    assert versions[0].date == datetime(2026, 6, 26, 12, 0)


@pytest.mark.asyncio
async def test_upsert_with_same_edit_date_records_previous_version(sqlite_adapter):
    edit_date = datetime(2026, 6, 26, 12, 30)
    await sqlite_adapter.insert_message(
        {
            "id": 33,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "original",
            "edit_date": edit_date,
        }
    )

    await sqlite_adapter.insert_message(
        {
            "id": 33,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "edited via backup",
            "edit_date": edit_date,
        },
        source="backup_upsert",
    )

    message = await _get_message(sqlite_adapter, 33, 300)
    versions = await _get_versions(sqlite_adapter, 33, 300)
    assert message.text == "edited via backup"
    assert message.edit_date == edit_date
    assert len(versions) == 1
    assert versions[0].text == "original"
    assert versions[0].date == edit_date


@pytest.mark.asyncio
async def test_upsert_with_same_aware_edit_date_records_previous_version(sqlite_adapter):
    edit_date = datetime(2026, 6, 26, 12, 30)
    aware_edit_date = datetime(2026, 6, 26, 12, 30, tzinfo=UTC)
    await sqlite_adapter.insert_message(
        {
            "id": 34,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "original",
            "edit_date": edit_date,
        }
    )

    await sqlite_adapter.insert_message(
        {
            "id": 34,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "edited via backup",
            "edit_date": aware_edit_date,
        },
        source="backup_upsert",
    )

    message = await _get_message(sqlite_adapter, 34, 300)
    versions = await _get_versions(sqlite_adapter, 34, 300)
    assert message.text == "edited via backup"
    assert message.edit_date == edit_date
    assert len(versions) == 1
    assert versions[0].text == "original"
    assert versions[0].date == edit_date


@pytest.mark.asyncio
async def test_repeated_upsert_with_same_edit_date_is_idempotent(sqlite_adapter):
    edit_date = datetime(2026, 6, 26, 12, 30)
    edited_message = {
        "id": 35,
        "chat_id": 300,
        "date": datetime(2026, 6, 26, 12, 0),
        "text": "edited via backup",
        "edit_date": edit_date,
    }
    await sqlite_adapter.insert_message(
        {
            "id": 35,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "original",
            "edit_date": edit_date,
        }
    )

    await sqlite_adapter.insert_message(edited_message, source="backup_upsert")
    await sqlite_adapter.insert_message(edited_message, source="backup_upsert")

    message = await _get_message(sqlite_adapter, 35, 300)
    versions = await _get_versions(sqlite_adapter, 35, 300)
    assert message.text == "edited via backup"
    assert message.edit_date == edit_date
    assert len(versions) == 1
    assert versions[0].text == "original"
    assert versions[0].date == edit_date


@pytest.mark.asyncio
async def test_upsert_same_text_updates_edit_date_without_version(sqlite_adapter):
    await sqlite_adapter.insert_message(
        {"id": 29, "chat_id": 300, "date": datetime(2026, 6, 26, 12, 0), "text": "original"}
    )
    reaction_edit_date = datetime(2026, 6, 26, 12, 30)

    await sqlite_adapter.insert_message(
        {
            "id": 29,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 12, 0),
            "text": "original",
            "edit_date": reaction_edit_date,
        },
        source="backup_upsert",
    )

    message = await _get_message(sqlite_adapter, 29, 300)
    versions = await _get_versions(sqlite_adapter, 29, 300)
    assert message.text == "original"
    assert message.edit_date == reaction_edit_date
    assert versions == []


@pytest.mark.asyncio
async def test_upsert_with_older_edit_date_does_not_roll_back(sqlite_adapter):
    current_edit_date = datetime(2026, 6, 26, 13, 30)
    old_edit_date = datetime(2026, 6, 26, 13, 5)
    await sqlite_adapter.insert_message(
        {
            "id": 24,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 13, 0),
            "text": "current text",
            "edit_date": current_edit_date,
        }
    )

    await sqlite_adapter.insert_message(
        {
            "id": 24,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 13, 0),
            "text": "old import text",
            "edit_date": old_edit_date,
        },
        source="import",
    )

    message = await _get_message(sqlite_adapter, 24, 300)
    versions = await _get_versions(sqlite_adapter, 24, 300)
    assert message.text == "current text"
    assert message.edit_date == current_edit_date
    assert versions == []


@pytest.mark.asyncio
async def test_concurrent_upserts_keep_newest_edit_date(sqlite_adapter):
    await sqlite_adapter.insert_message(
        {"id": 32, "chat_id": 300, "date": datetime(2026, 6, 26, 15, 0), "text": "original"}
    )

    async def upsert(text: str, edit_date: datetime) -> None:
        await sqlite_adapter.insert_message(
            {
                "id": 32,
                "chat_id": 300,
                "date": datetime(2026, 6, 26, 15, 0),
                "text": text,
                "edit_date": edit_date,
            },
            source="backup_upsert",
        )

    newer = datetime(2026, 6, 26, 15, 30)
    older = datetime(2026, 6, 26, 15, 10)
    await asyncio.gather(upsert("newer", newer), upsert("older", older))

    message = await _get_message(sqlite_adapter, 32, 300)
    assert message.text == "newer"
    assert message.edit_date == newer


@pytest.mark.asyncio
async def test_upsert_filling_empty_text_preserves_existing_edit_date(sqlite_adapter):
    current_edit_date = datetime(2026, 6, 26, 13, 45)
    await sqlite_adapter.insert_message(
        {
            "id": 26,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 13, 40),
            "text": "",
            "edit_date": current_edit_date,
        }
    )

    await sqlite_adapter.insert_message(
        {
            "id": 26,
            "chat_id": 300,
            "date": datetime(2026, 6, 26, 13, 40),
            "text": "filled text",
        },
        source="backup_upsert",
    )

    message = await _get_message(sqlite_adapter, 26, 300)
    versions = await _get_versions(sqlite_adapter, 26, 300)
    assert message.text == "filled text"
    assert message.edit_date == current_edit_date
    assert len(versions) == 1
    assert versions[0].text == ""
    assert versions[0].date == current_edit_date


@pytest.mark.asyncio
async def test_get_message_versions_returns_dicts(sqlite_adapter):
    await sqlite_adapter.insert_message({"id": 25, "chat_id": 300, "date": datetime(2026, 6, 26, 14, 0), "text": "v1"})
    await sqlite_adapter.update_message_text(300, 25, "v2", datetime(2026, 6, 26, 14, 5), source="sync_edit")
    await sqlite_adapter.update_message_text(300, 25, "v3", datetime(2026, 6, 26, 14, 10), source="sync_edit")

    versions = await sqlite_adapter.get_message_versions(300, 25)
    assert len(versions) == 2
    assert versions[0]["message_id"] == 25
    assert versions[0]["chat_id"] == 300
    assert "id" not in versions[0]
    assert "change_hash" not in versions[0]
    assert "captured_at" not in versions[0]
    assert [version["text"] for version in versions] == ["v2", "v1"]
    assert [version["date"] for version in versions] == [
        datetime(2026, 6, 26, 14, 5),
        datetime(2026, 6, 26, 14, 0),
    ]

    limited_versions = await sqlite_adapter.get_message_versions(300, 25, limit=1)
    assert [version["text"] for version in limited_versions] == ["v2"]


@pytest.mark.asyncio
async def test_get_messages_paginated_includes_version_counts(sqlite_adapter):
    await sqlite_adapter.insert_message({"id": 40, "chat_id": 300, "date": datetime(2026, 6, 26, 15, 0), "text": "v1"})
    await sqlite_adapter.insert_message(
        {"id": 41, "chat_id": 300, "date": datetime(2026, 6, 26, 15, 1), "text": "unchanged"}
    )
    await sqlite_adapter.update_message_text(300, 40, "v2", datetime(2026, 6, 26, 15, 5), source="sync_edit")
    await sqlite_adapter.update_message_text(300, 40, "v3", datetime(2026, 6, 26, 15, 10), source="sync_edit")

    messages = await sqlite_adapter.get_messages_paginated(300, limit=10)
    counts = {message["id"]: message["version_count"] for message in messages}
    assert counts[40] == 2
    assert counts[41] == 0


@pytest.mark.asyncio
async def test_get_message_versions_by_date_range_filters_version_dates(sqlite_adapter):
    await sqlite_adapter.insert_message({"id": 30, "chat_id": 300, "date": datetime(2026, 6, 25, 14, 0), "text": "old"})
    await sqlite_adapter.insert_message({"id": 31, "chat_id": 300, "date": datetime(2026, 6, 26, 14, 0), "text": "new"})
    await sqlite_adapter.update_message_text(300, 30, "old edited", datetime(2026, 6, 25, 14, 5))
    await sqlite_adapter.update_message_text(300, 31, "new edited", datetime(2026, 6, 26, 14, 5))

    versions = await sqlite_adapter.get_message_versions_by_date_range(
        chat_id=300,
        start_date=datetime(2026, 6, 26),
        end_date=datetime(2026, 6, 27),
    )

    assert [row["message_id"] for row in versions] == [31]
