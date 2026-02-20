from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_root_redirects_to_projects_page(test_client) -> None:
    response = await test_client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/projects"


@pytest.mark.asyncio
async def test_projects_page_renders_project_cards(test_client) -> None:
    await test_client.post(
        "/api/projects",
        json={"name": "GUI Project", "task": "classification"},
    )

    response = await test_client.get("/projects")

    assert response.status_code == 200
    assert "Projects" in response.text
    assert "GUI Project" in response.text
    assert "/projects/" in response.text
    assert "/dataset" in response.text


@pytest.mark.asyncio
async def test_dataset_page_shows_empty_state_for_selected_project(test_client) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Dataset Nav Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    response = await test_client.get(f"/projects/{project_id}/dataset")

    assert response.status_code == 200
    assert "No dataset imported." in response.text
    assert "Import Dataset" in response.text


@pytest.mark.asyncio
async def test_dataset_page_renders_initial_image_grid_after_import(
    test_client,
    workspace: Path,
) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Dataset Browser Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    source_root = workspace.parent / "dataset_page_source"
    (source_root / "cat").mkdir(parents=True, exist_ok=True)
    (source_root / "dog").mkdir(parents=True, exist_ok=True)
    _create_image(source_root / "cat" / "cat_page_001.png", color=(220, 120, 120))
    _create_image(source_root / "dog" / "dog_page_001.png", color=(120, 120, 220))

    imported = await test_client.post(
        f"/api/datasets/{project_id}/import/local",
        json={
            "source_path": str(source_root),
            "source_format": "image_folders",
        },
    )
    assert imported.status_code == 200

    response = await test_client.get(f"/projects/{project_id}/dataset")

    assert response.status_code == 200
    assert "cat_page_001.png" in response.text
    assert "dog_page_001.png" in response.text
    assert "Showing 1" in response.text
    assert f"/api/datasets/{project_id}/thumbnails/cat_page_001.png" in response.text
    assert f"/api/datasets/{project_id}/images" in response.text
    assert "/api/datasets//thumbnails/" not in response.text


@pytest.mark.asyncio
async def test_hx_project_create_rename_delete_flow(test_client) -> None:
    create_response = await test_client.post(
        "/api/projects",
        data={"name": "HTMX Project", "task": "classification"},
        headers={"HX-Request": "true"},
    )
    assert create_response.status_code == 200
    assert "HTMX Project" in create_response.text

    listed = await test_client.get("/api/projects")
    project_id = listed.json()["data"]["projects"][0]["id"]

    rename_response = await test_client.patch(
        f"/api/projects/{project_id}",
        data={"name": "HTMX Renamed"},
        headers={"HX-Request": "true"},
    )
    assert rename_response.status_code == 200
    assert "HTMX Renamed" in rename_response.text

    delete_response = await test_client.delete(
        f"/api/projects/{project_id}",
        headers={"HX-Request": "true"},
    )
    assert delete_response.status_code == 200
    assert "No projects yet." in delete_response.text

    final_list = await test_client.get("/api/projects")
    assert final_list.status_code == 200
    assert final_list.json()["data"]["projects"] == []


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size=(240, 180), color=color)
    image.save(path)
