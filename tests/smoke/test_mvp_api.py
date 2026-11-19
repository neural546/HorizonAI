def test_flask_starts_and_generate(flask_client):
    assert flask_client.get("/health").status_code == 200
    resp = flask_client.post("/generate", json={
        "prompt": "a simple test scene", "cfg_scale": 7.5, "num_frames": 8,
        "image_size": 64, "fps": 8, "seed": 0,
    })
    assert resp.status_code == 202
    data = resp.get_json()
    prog = flask_client.get(f"/progress?job_id={data['job_id']}")
    assert prog.status_code == 200
    assert prog.get_json()["status"] in ("queued", "running", "completed", "failed")
