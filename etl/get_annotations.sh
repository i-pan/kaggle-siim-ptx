PROJECT_ID="kaggle-siim-healthcare"
REGION="us-central1"
DATASET_ID="siim-pneumothorax"
FHIR_STORE_ID="fhir-masks-train"
DOCUMENT_REFERENCE_ID="d70d8f3e-990a-4bc0-b11f-c87349f5d4eb"

curl -X GET \
-H "Authorization: Bearer "$(gcloud auth print-access-token) \
"https://healthcare.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/datasets/${DATASET_ID}/fhirStores/${FHIR_STORE_ID}/fhir/DocumentReference/${DOCUMENT_REFERENCE_ID}"