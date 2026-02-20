from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_split_preview_create_list_get_delete_via_api(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Split API Project")
    source_root = workspace.parent / "split_api_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    import_response = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert import_response.status_code == 200

    initial_splits = await test_client.get(f"/api/splits/{project_id}")
    assert initial_splits.status_code == 200
    assert initial_splits.json()["data"]["splits"] == []

    preview_response = await test_client.get(
        f"/api/splits/{project_id}/preview",
        params={"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
    )
    assert preview_response.status_code == 200
    preview_payload = preview_response.json()["data"]
    assert preview_payload["stats"] == {"train": 32, "val": 4, "test": 4, "none": 0}
    assert preview_payload["class_distribution"]["cats"]["train"] == 16
    assert preview_payload["class_distribution"]["dogs"]["train"] == 16

    create_response = await test_client.post(
        f"/api/splits/{project_id}",
        json={
            "name": "80-10-10",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 42,
        },
    )
    assert create_response.status_code == 200
    created_payload = create_response.json()["data"]
    assert created_payload["name"] == "80-10-10"
    assert created_payload["index"] == 0
    assert created_payload["immutable"] is True

    list_response = await test_client.get(f"/api/splits/{project_id}")
    assert list_response.status_code == 200
    listed_splits = list_response.json()["data"]["splits"]
    assert len(listed_splits) == 1
    assert listed_splits[0]["name"] == "80-10-10"

    get_response = await test_client.get(f"/api/splits/{project_id}/80-10-10")
    assert get_response.status_code == 200
    assert get_response.json()["data"]["name"] == "80-10-10"

    delete_response = await test_client.delete(f"/api/splits/{project_id}/80-10-10")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "status": "ok",
        "data": {"project_id": project_id, "split_name": "80-10-10", "deleted": True},
    }

    list_after_delete = await test_client.get(f"/api/splits/{project_id}")
    assert list_after_delete.status_code == 200
    assert list_after_delete.json()["data"]["splits"] == []


@pytest.mark.asyncio
async def test_split_create_accepts_hx_form_payload(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Split HTMX Project")
    source_root = workspace.parent / "split_hx_source"
    _build_classification_source(source_root, cats=20, dogs=20)

    import_response = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert import_response.status_code == 200

    create_response = await test_client.post(
        f"/api/splits/{project_id}",
        data={
            "name": "70-20-10",
            "train": "0.70",
            "val": "0.20",
            "test": "0.10",
            "seed": "7",
        },
        headers={"HX-Request": "true"},
    )
    assert create_response.status_code == 200
    assert 'id="split-list"' in create_response.text
    assert "70-20-10" in create_response.text
    assert "Immutable" in create_response.text
    assert f"/api/splits/{project_id}/70-20-10" in create_response.text

    list_response = await test_client.get(f"/api/splits/{project_id}")
    assert list_response.status_code == 200
    listed_splits = list_response.json()["data"]["splits"]
    assert [split["name"] for split in listed_splits] == ["70-20-10"]


@pytest.mark.asyncio
async def test_split_preview_without_dataset_returns_standard_error(test_client) -> None:
    project_id = await _create_project(test_client, name="Split Error Project")

    response = await test_client.get(
        f"/api/splits/{project_id}/preview",
        params={"train": 0.8, "val": 0.1, "test": 0.1},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "DATASET_NOT_IMPORTED"


async def _create_project(test_client, *, name: str) -> str:
    response = await test_client.post(
        "/api/projects",
        json={
            "name": name,
            "task": "classification",
        },
    )
    return response.json()["data"]["id"]


def _build_classification_source(source_root: Path, *, cats: int, dogs: int) -> None:
    cats_dir = source_root / "cats"
    dogs_dir = source_root / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    for index in range(cats):
        _create_image(cats_dir / f"cat_{index:03d}.png", color=(220, 120, 120))
    for index in range(dogs):
        _create_image(dogs_dir / f"dog_{index:03d}.png", color=(120, 120, 220))


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(48, 48), color=color)
    image.save(path)
