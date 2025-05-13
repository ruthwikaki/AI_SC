# S3 bucket for model storage
resource "aws_s3_bucket" "model_storage" {
  bucket = "${var.project_name}-model-storage-${random_string.bucket_suffix.result}"
  
  tags = {
    Name = "${var.project_name}-model-storage"
  }
}

# Generate random suffix for globally unique bucket name
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 bucket for application assets
resource "aws_s3_bucket" "app_assets" {
  bucket = "${var.project_name}-assets-${random_string.bucket_suffix.result}"
  
  tags = {
    Name = "${var.project_name}-assets"
  }
}

# S3 bucket policy for model storage
resource "aws_s3_bucket_policy" "model_storage_policy" {
  bucket = aws_s3_bucket.model_storage.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
        ]
        Effect = "Allow"
        Principal = {
          AWS = aws_instance.ml_server.arn
        }
        Resource = [
          "${aws_s3_bucket.model_storage.arn}/*",
        ]
      }
    ]
  })
}

# Output the bucket names
output "model_storage_bucket" {
  value = aws_s3_bucket.model_storage.bucket
}

output "app_assets_bucket" {
  value = aws_s3_bucket.app_assets.bucket
}