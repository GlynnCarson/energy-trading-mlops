###############################################################################
# terraform/variables.tf
###############################################################################

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev | staging | prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Prefix for all resource names"
  type        = string
  default     = "energy-trading-mlops"
}

variable "data_lake_bucket" {
  description = "S3 bucket name for the data lake"
  type        = string
  default     = "energy-trading-mlops-data"
}

variable "glue_database" {
  description = "AWS Glue database name"
  type        = string
  default     = "energy_trading_db"
}

variable "alert_email" {
  description = "Email address for SNS pipeline failure alerts"
  type        = string
  sensitive   = true
}
