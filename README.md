# open — 드론 사진 UGV 검출 자동 학습 (MAICON 2026 M6 준비)

대회장 연습 시간에 **드론 사진을 자동으로 받고 → 자동으로 라벨을 달고 → 학습하고 → 로봇에 넣는** 파이프라인.
결과물은 로봇에서 도는 `best.onnx`(+ `model.json`)와 그걸 쓰는 `robot_check.py`.
경기 중에는 돌리지 않는다. Claude API 는 라벨 만들 때만(현장 PC), 로봇은 OpenCV 만 쓴다.

## 흐름

```
① 수집      서버 /api/report/drone 1초마다 → work/frames/       (같은 사진은 건너뜀)
② 자동 라벨  배경(중앙값) 차분 + 안테나 색 → 후보 → Claude 가 번호 상자 검증 → work/dataset/real/
③ 합성      우리 로봇 한 대를 잘라 2~6대 붙인 사진, 안테나 색 바꾸기 → work/dataset/synth/
④ 학습      YOLO11n 파인튜닝 → 시험(실사진만) mAP50 검사 → ONNX(opset 12) → work/models/vNNN/
⑤ 배포      검사 통과 시 로봇에 scp → 로봇에서 robot_check.py 로 열어 보고 한 장 추론
```
라벨이 늘면(100장 이상 & 30% 이상) 자동으로 다시 학습한다. 나빠진 모델은 배포하지 않는다.

## 처음 한 번 (현장 PC, Windows 11 · Python 3.10)

```powershell
git clone https://github.com/kim-109rhdqud/open
cd open
pip install -r requirements.txt
python run.py selftest          # 가짜 서버로 전체 흐름. "전부 통과" 가 나와야 함(GPU 면 1~2분)
copy local.example.yaml local.yaml   # 서버 주소·로봇 IP 적기
python run.py setup-key         # 로봇 비밀번호 한 번 → 이후 자동 배포
```

## 연습 시간

```powershell
$env:MAICON_CODE = "임무코드"            # 화면·파일에 안 남음
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # 없으면 OpenCV 만으로 라벨
python run.py ping        # 사진 한 장 → work/ping.jpg 로 UGV 크기·LED 보이는지 눈으로
python run.py pipeline    # 전부 자동. Ctrl-C 로 멈추고 다시 켜면 이어서
```
- 그동안 우리 로봇을 경기장 전체(가장자리·상자 옆·여러 방향)로 몰고 다닌다. 멈춰 있으면 배울 게 없다.
- `local.yaml` 의 `label.robots_in_arena` = 경기장 안 UGV 수(연습 때 우리 1대면 1).
- 판정 그림: `work/review/ok`, `work/review/rejected`. 상태: `python run.py status`, `work/status.json`, 로그 `work/pipeline.log`.

## 결과물

| 위치 | 내용 |
|---|---|
| `work/models/current.json` | 지금 배포 대상 버전·mAP50 |
| `work/models/vNNN/best.pt` | 원본(다시 학습할 때) |
| `work/models/vNNN/best.onnx`, `model.json` | **로봇용** (입력 크기·클래스) |
| 로봇 `~/m6_model/` | `best.onnx`, `model.json`, `robot_check.py`, `sample.jpg`, 이전 `best_prev.onnx` |

로봇 코드에서 쓰기:
```python
from robot_check import load, detect
net, meta = load("/home/<user>/m6_model/best.onnx")
dets = detect(net, bgr_image, meta)   # [{'box':[x1,y1,x2,y2],'conf':..,'team':'RED'|'BLUE'|None}]
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `run.py` | 명령 모음(selftest · ping · setup-key · pipeline · status · label · train · deploy) |
| `config.yaml` | 기본 설정(고치지 말고 `local.yaml` 에 덮어쓰기) |
| `robot_check.py` | 로봇용 검출·점검(ONNX + OpenCV DNN, PyTorch 불필요) |
| `dronecv/collect.py` | 수집 |
| `dronecv/vision.py` | OpenCV 후보(배경 차분·안테나 색) |
| `dronecv/claude.py` | Claude 검증(키는 헤더로만, 기록 안 함) |
| `dronecv/autolabel.py` | 판정 규칙·라벨 저장 |
| `dronecv/augment.py` | 잘라 붙이기·색 바꾸기 |
| `dronecv/dataset.py` | 학습·시험 목록 |
| `dronecv/train.py` | 학습·검사·ONNX |
| `dronecv/deploy.py` | 로봇 복사·점검, ssh 키 등록 |
| `dronecv/pipeline.py` | 자동 반복 |
| `dronecv/synth.py`, `fakeserver.py` | 시험용 가짜 사진·서버 |

## 시험

```
python -m pytest -q tests            # 빠른 시험(학습 없음)
RUN_SLOW=1 python -m pytest -q tests # 학습 포함
```

## 확인한 것 / 못 한 것

- 확인(리눅스, CPU): 가짜 서버 → 수집 → 라벨 → 합성 → 학습 → ONNX → OpenCV DNN 점검 → 로컬 배포. 우리 로봇 1대 사진만으로 학습한 모델이 가짜 4대 사진에서 39/40 검출, 팀 색 39/39.
- 못 함: 실제 드론 사진(가짜 사진은 실물과 다름), 실제 Claude API 호출(요청 형식은 가짜로 시험), Windows 실행, 로봇 ssh 배포.
- 정해지지 않은 것: 연습 기간에 개인 Claude 키를 써도 되는지(11/9 답변: 준비 기간은 허용), 온보드 OpenCV/YOLO 허용 여부(Q&A 대기).
