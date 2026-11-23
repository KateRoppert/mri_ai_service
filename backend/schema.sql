CREATE TABLE pipeline_runs (
	run_id VARCHAR NOT NULL, 
	input_path VARCHAR NOT NULL, 
	output_path VARCHAR NOT NULL, 
	status VARCHAR NOT NULL, 
	current_stage INTEGER, 
	overall_progress FLOAT, 
	quality_score FLOAT, 
	quality_category VARCHAR, 
	created_at DATETIME, 
	started_at DATETIME, 
	completed_at DATETIME, 
	error_message TEXT, 
	config_path VARCHAR, 
	lesion_type VARCHAR, 
	kappa_dataset_id INTEGER, 
	preprocessing_id VARCHAR, 
	stopped_at_stage INTEGER, 
	stopped_by VARCHAR, 
	parent_run_id VARCHAR, 
	kappa_upload_status VARCHAR, 
	kappa_upload_next_attempt DATETIME, 
	kappa_upload_detail TEXT, 
	kappa_user_id INTEGER, 
	reprocessed_sessions TEXT, 
	queued_requeue_at DATETIME, 
	PRIMARY KEY (run_id)
);
CREATE INDEX ix_pipeline_runs_run_id ON pipeline_runs (run_id);
CREATE TABLE stage_executions (
	id INTEGER NOT NULL, 
	run_id VARCHAR NOT NULL, 
	stage_number INTEGER NOT NULL, 
	stage_name VARCHAR NOT NULL, 
	status VARCHAR NOT NULL, 
	progress FLOAT, 
	started_at DATETIME, 
	completed_at DATETIME, 
	error_message TEXT, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_stage_executions_run_id ON stage_executions (run_id);
CREATE TABLE kappa_sessions (
	session_id VARCHAR NOT NULL, 
	kappa_token VARCHAR NOT NULL, 
	user_id INTEGER, 
	user_type_id INTEGER, 
	user_name VARCHAR, 
	first_name VARCHAR, 
	last_name VARCHAR, 
	token_expiry VARCHAR, 
	org_details TEXT, 
	created_at DATETIME, 
	PRIMARY KEY (session_id)
);
CREATE TABLE patient_registry (
	id INTEGER NOT NULL, 
	study_hash VARCHAR NOT NULL, 
	bids_id VARCHAR NOT NULL, 
	original_patient_id VARCHAR NOT NULL, 
	patient_name VARCHAR, 
	scan_date VARCHAR, 
	study_instance_uid VARCHAR, 
	kappa_entity_id VARCHAR, 
	kappa_dataset_id INTEGER, 
	pipeline_run_id VARCHAR, 
	lesion_type VARCHAR, 
	preprocessing_id VARCHAR, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_patient_registry_original_patient_id ON patient_registry (original_patient_id);
CREATE INDEX ix_patient_registry_kappa_dataset_id ON patient_registry (kappa_dataset_id);
CREATE INDEX ix_patient_registry_bids_id ON patient_registry (bids_id);
CREATE UNIQUE INDEX ix_patient_registry_study_dataset ON patient_registry (study_hash, COALESCE(kappa_dataset_id, 0));
CREATE INDEX ix_patient_registry_pipeline_run_id ON patient_registry (pipeline_run_id);
CREATE INDEX ix_patient_registry_study_hash ON patient_registry (study_hash);
CREATE INDEX ix_patient_registry_patient_date ON patient_registry (original_patient_id, scan_date);
CREATE INDEX ix_patient_registry_kappa_entity_id ON patient_registry (kappa_entity_id);
CREATE TABLE validations (
	id INTEGER NOT NULL, 
	entity_id VARCHAR NOT NULL, 
	dataset_id INTEGER NOT NULL, 
	user_id INTEGER NOT NULL, 
	user_name VARCHAR, 
	action VARCHAR NOT NULL, 
	comment TEXT, 
	created_at DATETIME, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_validations_user_id ON validations (user_id);
CREATE INDEX ix_validations_entity_id ON validations (entity_id);
CREATE INDEX ix_validations_dataset_id ON validations (dataset_id);
CREATE INDEX ix_validations_created_at ON validations (created_at);
CREATE INDEX ix_validations_entity_user ON validations (entity_id, user_id);
CREATE TABLE mask_versions (
	id INTEGER NOT NULL, 
	entity_id VARCHAR NOT NULL, 
	dataset_id INTEGER NOT NULL, 
	version INTEGER NOT NULL, 
	source VARCHAR NOT NULL, 
	uploaded_by_user_id INTEGER, 
	uploaded_by_name VARCHAR, 
	file_path VARCHAR NOT NULL, 
	file_name VARCHAR NOT NULL, 
	kappa_file_id VARCHAR, 
	created_at DATETIME, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_mask_versions_entity_version ON mask_versions (entity_id, version);
CREATE INDEX ix_mask_versions_created_at ON mask_versions (created_at);
CREATE INDEX ix_mask_versions_entity_id ON mask_versions (entity_id);
CREATE TABLE preprocessing_versions (
	preprocessing_id VARCHAR NOT NULL, 
	created_at DATETIME, 
	config_json TEXT NOT NULL, 
	PRIMARY KEY (preprocessing_id)
);
