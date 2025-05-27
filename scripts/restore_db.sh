#!/bin/bash
# scripts/backup_db.sh

# SSA Database Backup Script
# This script creates a backup of the PostgreSQL database

set -e  # Exit on error

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Parse DATABASE_URL
# Format: postgresql://user:password@host:port/database
DB_URL=${DATABASE_URL}
DB_USER=$(echo $DB_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p')
DB_PASS=$(echo $DB_URL | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
DB_HOST=$(echo $DB_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
DB_PORT=$(echo $DB_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
DB_NAME=$(echo $DB_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')

# Default values
DB_PORT=${DB_PORT:-5432}
BACKUP_DIR=${BACKUP_DIR:-"./backups"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup types
BACKUP_TYPE=${1:-"full"}  # full, schema, or data

# Create backup directory
mkdir -p $BACKUP_DIR

# Function to print colored output
print_info() {
    echo -e "\033[0;36m$1\033[0m"
}

print_success() {
    echo -e "\033[0;32m$1\033[0m"
}

print_error() {
    echo -e "\033[0;31m$1\033[0m"
}

# Backup file names
if [ "$BACKUP_TYPE" = "schema" ]; then
    BACKUP_FILE="$BACKUP_DIR/ssa_schema_${TIMESTAMP}.sql"
    BACKUP_CUSTOM="$BACKUP_DIR/ssa_schema_${TIMESTAMP}.dump"
elif [ "$BACKUP_TYPE" = "data" ]; then
    BACKUP_FILE="$BACKUP_DIR/ssa_data_${TIMESTAMP}.sql"
    BACKUP_CUSTOM="$BACKUP_DIR/ssa_data_${TIMESTAMP}.dump"
else
    BACKUP_FILE="$BACKUP_DIR/ssa_full_${TIMESTAMP}.sql"
    BACKUP_CUSTOM="$BACKUP_DIR/ssa_full_${TIMESTAMP}.dump"
fi

BACKUP_COMPRESSED="$BACKUP_FILE.gz"

print_info "Starting database backup..."
print_info "Database: $DB_NAME"
print_info "Host: $DB_HOST:$DB_PORT"
print_info "Backup type: $BACKUP_TYPE"

# Export password for pg_dump
export PGPASSWORD=$DB_PASS

# Perform backup based on type
case $BACKUP_TYPE in
    "schema")
        print_info "Creating schema-only backup..."
        pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
            --schema-only \
            --no-owner \
            --no-privileges \
            --if-exists \
            --clean \
            -f $BACKUP_FILE
        
        # Also create custom format for faster restore
        pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
            --schema-only \
            --no-owner \
            --no-privileges \
            --format=custom \
            -f $BACKUP_CUSTOM
        ;;
    
    "data")
        print_info "Creating data-only backup..."
        pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
            --data-only \
            --disable-triggers \
            -f $BACKUP_FILE
        
        # Custom format
        pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
            --data-only \
            --disable-triggers \
            --format=custom \
            -f $BACKUP_CUSTOM
        ;;
    
    "full")
        print_info "Creating full backup..."
        pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
            --no-owner \
            --no-privileges \
            --if-exists \
            --clean \
            -f $BACKUP_FILE
        
        # Custom format for faster restore
        pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
            --no-owner \
            --no-privileges \
            --format=custom \
            -f $BACKUP_CUSTOM
        ;;
    
    *)
        print_error "Invalid backup type: $BACKUP_TYPE"
        print_info "Usage: $0 [full|schema|data]"
        exit 1
        ;;
esac

# Check if backup was successful
if [ $? -eq 0 ]; then
    print_success "Backup created successfully!"
    
    # Compress the SQL backup
    print_info "Compressing backup..."
    gzip -c $BACKUP_FILE > $BACKUP_COMPRESSED
    
    # Remove uncompressed SQL file to save space
    rm $BACKUP_FILE
    
    # Show backup info
    print_info "Backup files created:"
    print_info "  - Compressed SQL: $BACKUP_COMPRESSED"
    print_info "  - Custom format: $BACKUP_CUSTOM"
    
    # Show file sizes
    print_info "Backup sizes:"
    ls -lh $BACKUP_COMPRESSED $BACKUP_CUSTOM
    
    # Clean old backups (keep last 7 days)
    if [ "$CLEANUP_OLD_BACKUPS" = "true" ]; then
        print_info "Cleaning old backups..."
        find $BACKUP_DIR -name "ssa_*.sql.gz" -mtime +7 -delete
        find $BACKUP_DIR -name "ssa_*.dump" -mtime +7 -delete
        print_success "Old backups cleaned"
    fi
    
    # Upload to S3 if configured
    if [ ! -z "$S3_BACKUP_BUCKET" ]; then
        print_info "Uploading to S3..."
        aws s3 cp $BACKUP_COMPRESSED s3://$S3_BACKUP_BUCKET/database/
        aws s3 cp $BACKUP_CUSTOM s3://$S3_BACKUP_BUCKET/database/
        print_success "Backup uploaded to S3"
    fi
    
else
    print_error "Backup failed!"
    exit 1
fi

# Unset password
unset PGPASSWORD

print_success "Backup completed successfully!"