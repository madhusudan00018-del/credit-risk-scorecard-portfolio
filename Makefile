.PHONY: install download pipeline mart dashboard test clean

install:
	pip install -r requirements.txt

download:
	python src/utils/download_data.py

pipeline:
	python src/utils/run_pipeline.py --config config/model_config.yaml

mart:
	python src/utils/build_sqlite_mart.py

dashboard:
	streamlit run app.py

test:
	pytest

clean:
	rm -f data/processed/*.csv data/processed/*.sqlite models/*.joblib models/*.json images/*.png

