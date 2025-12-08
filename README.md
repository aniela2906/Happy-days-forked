# ITU BDS MLOPS'25 - Project

## Chronological flow of our MLOps pipeline

### Phase 1: Initiation and data preparation 

The process begins when Github Actions Workflow is triggered.

The build-deploy job starts on Github hosted runner.

The runner then executes a dvc pull to download the actual training data file from the dvc demote store into the notebooks/artifacts folder. The runner makes a copy of the raw_data.csv into the Dagger input directory go/python-files/artifacts

The runner then executes the dagger run go run pipeline.go command, handing control over to the Dagger CLI.

### Phase 2: Training and Artifcat Creation (Dagger container)

The execution now moves inside the Docker environment defined by our pipeline.go script.

The dagger engine starts, pulls the base Python image and mounts our go/python-files to the container's working directory.

It then executes install some dependencies from requirements.txt and executes the training script train_model.py

The python script reads the staged data file and throught the script, calls to MLflow to log the metadata, metrics, parameters, the generated model files to the MLflow artifact store.

The script writes the final run id and stores it in mflow_run_id.txt and exports the artifacts.

### Phase 3: Inference validation test and finalization

The control returns to the Github runner to finalize the first job and execute the second job which is inference-test.

As the build-deploy job completes succesfully, the test-inference job starts.

The runner downloads the model artifact from Github's storage into its local directory.
Then it executes the test_inference.py script to validate the generated artifacts.

The test-inference job reports either succes or failure, and the entire MLops pipeline workflow finishes. 
