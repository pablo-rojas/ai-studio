from __future__ import annotations

import pytest


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
async def test_dataset_page_placeholder_loads_for_selected_project(test_client) -> None:
    created = await test_client.post(
        "/api/projects",
        json={"name": "Dataset Nav Project", "task": "classification"},
    )
    project_id = created.json()["data"]["id"]

    response = await test_client.get(f"/projects/{project_id}/dataset")

    assert response.status_code == 200
    assert "Dataset page for" in response.text
    assert "Phase 6" in response.text


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
