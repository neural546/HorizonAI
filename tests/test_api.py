def test_index(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200 and b"Text-to-Video" in resp.data


def test_static_css(flask_client):
    resp = flask_client.get("/static/style.css")
    assert resp.status_code == 200 and b"--accent" in resp.data


def test_static_js(flask_client):
    resp = flask_client.get("/static/app.js")
    assert resp.status_code == 200 and b"generate" in resp.data


def test_health(flask_client):
    resp = flask_client.get("/health")
    assert resp.status_code == 200 and resp.get_json()["status"] == "ok"


def test_generate_requires_prompt(flask_client):
    assert flask_client.post("/generate", json={}).status_code == 400


def test_generate_validates_cfg(flask_client):
    assert flask_client.post("/generate", json={"prompt": "a cat", "cfg_scale": 100.0}).status_code == 400


def test_generate_validates_frames(flask_client):
    assert flask_client.post("/generate", json={"prompt": "a cat", "num_frames": 2}).status_code == 400


def test_generate_validates_resolution(flask_client):
    assert flask_client.post("/generate", json={"prompt": "a cat", "image_size": 999}).status_code == 400


def test_generate_validates_numeric(flask_client):
    assert flask_client.post("/generate", json={"prompt": "a cat", "cfg_scale": "abc"}).status_code == 400


def test_progress_requires_job_id(flask_client):
    assert flask_client.get("/progress").status_code == 400


def test_progress_unknown_job(flask_client):
    assert flask_client.get("/progress?job_id=doesnotexist").status_code == 404


def test_generate_accepts_valid(flask_client):
    resp = flask_client.post("/generate", json={
        "prompt": "a cat walking", "cfg_scale": 7.5, "num_frames": 8,
        "image_size": 64, "fps": 8, "seed": 42,
    })
    assert resp.status_code == 202
    data = resp.get_json()
    assert "job_id" in data and data["status"] == "queued"
