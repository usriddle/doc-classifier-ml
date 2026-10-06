"""
⑤ Gemma QLoRA 학습용 패키지.

  records.py   정답 레코드(민원 문장 + 정답 의도/카테고리/도구 인자) 읽기·검증·분할
  samples.py   정답 레코드 -> 학습 샘플(토큰 id + loss 마스크). 추론과 같은 함수로 조립
  evaluate.py  운영 코드(llm_engine)로 모델을 평가하고 학습 전후를 비교
  trainer.py   LoRA 부착·학습 루프·실시간 그래프·중간 점검·저장

colab_train.ipynb 가 이 모듈들을 순서대로 부릅니다.
"""
