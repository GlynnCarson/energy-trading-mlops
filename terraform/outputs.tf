###############################################################################
# terraform/outputs.tf
###############################################################################

output "data_lake_bucket_name" {
  description = "S3 data lake bucket name"
  value       = aws_s3_bucket.data_lake.bucket
}

output "data_lake_bucket_arn" {
  description = "S3 data lake bucket ARN"
  value       = aws_s3_bucket.data_lake.arn
}

output "glue_database_name" {
  description = "Glue catalog database name"
  value       = aws_glue_catalog_database.energy_db.name
}

output "sagemaker_role_arn" {
  description = "SageMaker IAM role ARN (use in pipeline.yaml)"
  value       = aws_iam_role.sagemaker_role.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL for pipeline Docker images"
  value       = aws_ecr_repository.pipeline_image.repository_url
}

output "sns_topic_arn" {
  description = "SNS topic ARN for pipeline alerts"
  value       = aws_sns_topic.pipeline_alerts.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for pipeline logs"
  value       = aws_cloudwatch_log_group.pipeline_logs.name
}
