provider "aws" {
  region = var.aws_region
}

# Security group for EC2 instances
resource "aws_security_group" "app_security_group" {
  name        = "${var.project_name}-sg"
  description = "Security group for Supply Chain LLM application"

  # Web access
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # HTTPS access
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_ip_cidr]
  }
  
  # Application port
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # ML inference port
  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# EC2 instance for backend + frontend
resource "aws_instance" "app_server" {
  ami           = var.app_ami
  instance_type = var.app_instance_type
  key_name      = var.key_name
  
  vpc_security_group_ids = [aws_security_group.app_security_group.id]
  
  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }
  
  tags = {
    Name = "${var.project_name}-app-server"
  }
}

# EC2 instance for ML inference (larger for GPU)
resource "aws_instance" "ml_server" {
  ami           = var.ml_ami
  instance_type = var.ml_instance_type
  key_name      = var.key_name
  
  vpc_security_group_ids = [aws_security_group.app_security_group.id]
  
  root_block_device {
    volume_size = 100  # Larger for model storage
    volume_type = "gp3"
  }
  
  tags = {
    Name = "${var.project_name}-ml-server"
  }
}

# Elastic IP for app server
resource "aws_eip" "app_eip" {
  instance = aws_instance.app_server.id
  domain   = "vpc"
  
  tags = {
    Name = "${var.project_name}-app-eip"
  }
}

# Elastic IP for ML server
resource "aws_eip" "ml_eip" {
  instance = aws_instance.ml_server.id
  domain   = "vpc"
  
  tags = {
    Name = "${var.project_name}-ml-eip"
  }
}

# Output the public IPs
output "app_server_public_ip" {
  value = aws_eip.app_eip.public_ip
}

output "ml_server_public_ip" {
  value = aws_eip.ml_eip.public_ip
}