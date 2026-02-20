from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_dataset_import_and_browse_via_api(test_client, workspace: Path) -> None:
    project_id = await _create_project(test_client, name="Dataset API Project")
    source_root = workspace.parent / "dataset_api_source"
    (source_root / "cats").mkdir(parents=True, exist_ok=True)
    (source_root / "dogs").mkdir(parents=True, exist_ok=True)

    _create_image(source_root / "cats" / "cat_001.png", size=(300, 200), color=(220, 120, 120))
    _create_image(source_root / "cats" / "cat_002.png", size=(260, 180), color=(220, 90, 90))
    _create_image(source_root / "dogs" / "dog_001.png", size=(280, 210), color=(120, 120, 220))

    import_response = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert import_response.status_code == 200
    import_payload = import_response.json()
    assert import_payload["status"] == "ok"
    assert import_payload["data"]["image_stats"]["num_images"] == 3

    dataset_response = await test_client.get(f"/api/datasets/{project_id}")
    assert dataset_response.status_code == 200
    dataset_payload = dataset_response.json()
    assert dataset_payload["status"] == "ok"
    assert dataset_payload["data"]["classes"] == ["cats", "dogs"]

    images_response = await test_client.get(
        f"/api/datasets/{project_id}/images",
        params={"page": 1, "page_size": 2, "sort_by": "filename", "sort_order": "asc"},
    )
    assert images_response.status_code == 200
    images_payload = images_response.json()["data"]
    assert images_payload["total_items"] == 3
    assert images_payload["page"] == 1
    assert images_payload["page_size"] == 2
    assert len(images_payload["items"]) == 2

    large_page_response = await test_client.get(
        f"/api/datasets/{project_id}/images",
        params={"page": 1, "page_size": 500},
    )
    assert large_page_response.status_code == 200
    assert large_page_response.json()["data"]["page_size"] == 500

    cats_only_response = await test_client.get(
        f"/api/datasets/{project_id}/images",
        params={"filter_class": "cats"},
    )
    assert cats_only_response.status_code == 200
    cats_items = cats_only_response.json()["data"]["items"]
    assert cats_items
    assert all(item["class_name"] == "cats" for item in cats_items)

    target_filename = images_payload["items"][0]["filename"]
    image_info_response = await test_client.get(
        f"/api/datasets/{project_id}/images/{target_filename}/info"
    )
    assert image_info_response.status_code == 200
    assert image_info_response.json()["data"]["image"]["filename"] == target_filename

    image_response = await test_client.get(f"/api/datasets/{project_id}/images/{target_filename}")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/")

    thumbnail_response = await test_client.get(
        f"/api/datasets/{project_id}/thumbnails/{target_filename}"
    )
    assert thumbnail_response.status_code == 200
    assert thumbnail_response.headers["content-type"].startswith("image/")

    clear_response = await test_client.delete(f"/api/datasets/{project_id}")
    assert clear_response.status_code == 200
    assert clear_response.json() == {
        "status": "ok",
        "data": {"project_id": project_id, "cleared": True},
    }

    missing_dataset_response = await test_client.get(f"/api/datasets/{project_id}")
    assert missing_dataset_response.status_code == 404
    missing_payload = missing_dataset_response.json()
    assert missing_payload["status"] == "error"
    assert missing_payload["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_hx_dataset_image_listing_supports_filters_and_pagination(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Dataset HTMX Listing")
    source_root = workspace.parent / "dataset_hx_listing_source"
    (source_root / "cats").mkdir(parents=True, exist_ok=True)
    (source_root / "dogs").mkdir(parents=True, exist_ok=True)

    _create_image(source_root / "cats" / "cat_hx_001.png", size=(300, 200), color=(220, 120, 120))
    _create_image(source_root / "cats" / "cat_hx_002.png", size=(260, 180), color=(220, 90, 90))
    _create_image(source_root / "dogs" / "dog_hx_001.png", size=(280, 210), color=(120, 120, 220))

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    page_response = await test_client.get(
        f"/api/datasets/{project_id}/images",
        params={
            "page": 1,
            "page_size": 1,
            "sort_by": "filename",
            "sort_order": "asc",
        },
        headers={"HX-Request": "true"},
    )
    assert page_response.status_code == 200
    assert 'id="dataset-image-list"' in page_response.text
    assert 'id="dataset-page-input-top"' in page_response.text
    assert 'id="dataset-page-input-bottom"' in page_response.text
    assert page_response.text.count("Previous") >= 2
    assert page_response.text.count("Next") >= 2
    assert page_response.text.count("of 3") >= 2
    assert "cat_hx_001.png" in page_response.text

    filtered_response = await test_client.get(
        f"/api/datasets/{project_id}/images",
        params={"filter_class": "cats"},
        headers={"HX-Request": "true"},
    )
    assert filtered_response.status_code == 200
    assert "cat_hx_001.png" in filtered_response.text
    assert "cat_hx_002.png" in filtered_response.text
    assert "dog_hx_001.png" not in filtered_response.text


@pytest.mark.asyncio
async def test_hx_dataset_image_detail_fragment_renders_metadata(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Dataset HTMX Detail")
    source_root = workspace.parent / "dataset_hx_detail_source"
    (source_root / "cats").mkdir(parents=True, exist_ok=True)
    _create_image(
        source_root / "cats" / "cat_detail_001.png", size=(320, 240), color=(220, 120, 120)
    )

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    listing_response = await test_client.get(f"/api/datasets/{project_id}/images")
    assert listing_response.status_code == 200
    filename = listing_response.json()["data"]["items"][0]["filename"]

    detail_response = await test_client.get(
        f"/api/datasets/{project_id}/images/{filename}/info",
        headers={"HX-Request": "true"},
    )
    assert detail_response.status_code == 200
    assert filename in detail_response.text
    assert "Dimensions" in detail_response.text
    assert "Class" in detail_response.text


@pytest.mark.asyncio
async def test_hx_local_import_accepts_form_payload(
    test_client,
    workspace: Path,
) -> None:
    project_id = await _create_project(test_client, name="Dataset HTMX Form Import")
    source_root = workspace.parent / "dataset_hx_form_source"
    (source_root / "cats").mkdir(parents=True, exist_ok=True)
    _create_image(source_root / "cats" / "cat_form_001.png", size=(256, 192), color=(220, 120, 120))

    response = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        data={
            "source_path": str(source_root),
            "source_format": "",
        },
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert 'id="dataset-summary"' in response.text
    assert "No dataset imported." not in response.text


async def _create_project(test_client, *, name: str) -> str:
    response = await test_client.post(
        "/api/projects",
        json={
            "name": name,
            "task": "classification",
        },
    )
    return response.json()["data"]["id"]


def _create_image(
    path: Path,
    *,
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    image = Image.new("RGB", size=size, color=color)
    image.save(path)
