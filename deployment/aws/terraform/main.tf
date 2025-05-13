terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "supply-chain-llm"
}

variable "app_ami" {
  description = "AMI ID for app server"
  type        = string
  default     = "ami-0c65adc9a5c1b5d7c"  # Ubuntu 22.04 LTS
}

variable "ml_ami" {
  description = "AMI ID for ML server"
  type        = string
  default     = "ami-0c65adc9a5c1b5d7c"  # Ubuntu 22.04 LTS
}

variable "app_instance_type" {
  description = "Instance type for app server"
  type        = string
  default     = "t3.large"
}

variable "ml_instance_type" {
  description = "Instance type for ML server"
  type        = string
  default     = "g4dn.xlarge"  # GPU instance for ML inference
}

variable "key_name" {
  description = "EC2 key pair name"
  type        = string
}

variable "admin_ip_cidr" {
  description = "CIDR block for admin access"
  type        = string
  default     = "0.0.0.0/0"  # Should be restricted in production
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "supplychainllm"
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "subnet_ids" {
  description = "Subnet IDs for the DB subnet group"
  type        = list(string)
}