from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_projects_crud_via_api(test_client) -> None:
    list_response = await test_client.get("/api/projects")
    assert list_response.status_code == 200
    assert list_response.json() == {"status": "ok", "data": {"projects": []}}

    create_response = await test_client.post(
        "/api/projects",
        json={
            "name": "API Project",
            "task": "classification",
            "description": "Created through API",
        },
    )
    assert create_response.status_code == 200
    created_payload = create_response.json()
    created_project = created_payload["data"]
    project_id = created_project["id"]
    assert created_payload["status"] == "ok"
    assert project_id.startswith("proj-")
    assert created_project["name"] == "API Project"
    assert created_project["task"] == "classification"

    get_response = await test_client.get(f"/api/projects/{project_id}")
    assert get_response.status_code == 200
    assert get_response.json()["data"]["id"] == project_id

    rename_response = await test_client.patch(
        f"/api/projects/{project_id}",
        json={"name": "API Project Renamed"},
    )
    assert rename_response.status_code == 200
    assert rename_response.json()["data"]["name"] == "API Project Renamed"

    delete_response = await test_client.delete(f"/api/projects/{project_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "status": "ok",
        "data": {"project_id": project_id, "deleted": True},
    }

    missing_response = await test_client.get(f"/api/projects/{project_id}")
    assert missing_response.status_code == 404
    missing_payload = missing_response.json()
    assert missing_payload["status"] == "error"
    assert missing_payload["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_project_validation_errors_use_standard_error_envelope(test_client) -> None:
    response = await test_client.post("/api/projects", json={"name": ""})

    assert response.status_code == 422
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["message"] == "Request validation failed."
    assert isinstance(payload["error"]["details"], list)
