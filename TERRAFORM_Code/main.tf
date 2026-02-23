provider "aws" {
  region = "us-east-1"
}

# import default vpc
data "aws_vpc" "default" {
  default = true
}

# import default subnets
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# IAM Role
resource "aws_iam_role" "aiops_role" {
  name               = "aiops-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "aiops-ec2-role"
  }
}

# IAM Policy - CloudWatch Agent Server Policy
resource "aws_iam_role_policy_attachment" "cloudwatch_agent_server_policy" {
  role       = aws_iam_role.aiops_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# IAM Policy - SSM Managed Instance Core
resource "aws_iam_role_policy_attachment" "ssm_managed_instance_core" {
  role       = aws_iam_role.aiops_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "aiops_instance_profile" {
  name = "aiops-instance-profile"
  role = aws_iam_role.aiops_role.name
}

# create security group for EC2 instance
resource "aws_security_group" "aiops_sg" {
  name        = "aiops-security-group"
  description = "Security group for AIOps EC2 instance"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTP access"
    from_port   = 80
    to_port     = 80
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
        Name = "aiops-security-group"
    }
}

# Get latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux_2" {   
  most_recent = true
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["amazon"]
}

# RSA key pair
resource "tls_private_key" "aiops_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# create local file to store private key
resource "local_file" "aiops_private_key" {
  content  = tls_private_key.aiops_key.private_key_pem
  filename = "aiops_key.pem"
  file_permission = "0400"
}

# create public key in AWS
resource "aws_key_pair" "aiops_key_pair" {
  key_name   = "aiops-key-pair"
  public_key = tls_private_key.aiops_key.public_key_openssh
}

# EC2 Instance
resource "aws_instance" "aiops_instance" {
  ami                         = data.aws_ami.amazon_linux_2.id
  instance_type               = "t2.medium"
  iam_instance_profile        = aws_iam_instance_profile.aiops_instance_profile.name
  key_name                    = aws_key_pair.aiops_key_pair.key_name
  vpc_security_group_ids      = [aws_security_group.aiops_sg.id]
  subnet_id                   = element(data.aws_subnets.default.ids, 0)
  associate_public_ip_address = true
  user_data                   = file("${path.module}/userdata.sh")
  credit_specification {
    cpu_credits = "unlimited"
  }

  tags = {
    Name = "aiops-ec2-instance"
    AnomalyMonitoring = "enabled"
  }
}

# Output the public IP of the EC2 instance
output "ec2_instance_public_ip" {
  value = aws_instance.aiops_instance.public_ip
}

# Output the id of the EC2 instance
output "ec2_instance_id" {
  value = aws_instance.aiops_instance.id
}