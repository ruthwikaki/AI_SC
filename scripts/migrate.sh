#!/bin/bash
# scripts/migrate.sh

# SSA Database Migration Script
# This script handles database migrations using Alembic

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

# Check if we're in the right directory
if [ ! -f "app/db/migrations/alembic.ini" ]; then
    print_error "Error: alembic.ini not found. Are you in the project root?"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    print_info "Environment variables loaded"
fi

# Parse command
COMMAND=${1:-"help"}

# Change to app directory for alembic
cd app/db/migrations

case $COMMAND in
    "create")
        # Create a new migration
        MESSAGE=$2
        if [ -z "$MESSAGE" ]; then
            print_error "Error: Migration message required"
            print_info "Usage: $0 create \"migration message\""
            exit 1
        fi
        
        print_info "Creating new migration: $MESSAGE"
        alembic revision --autogenerate -m "$MESSAGE"
        
        if [ $? -eq 0 ]; then
            print_success "Migration created successfully!"
            print_info "Review the generated migration file before applying"
        fi
        ;;
    
    "up"|"upgrade")
        # Upgrade to latest or specific revision
        TARGET=${2:-"head"}
        print_info "Upgrading database to: $TARGET"
        
        # Show current revision
        print_info "Current revision:"
        alembic current
        
        # Perform upgrade
        alembic upgrade $TARGET
        
        if [ $? -eq 0 ]; then
            print_success "Database upgraded successfully!"
            print_info "New revision:"
            alembic current
        fi
        ;;
    
    "down"|"downgrade")
        # Downgrade to previous or specific revision
        TARGET=${2:-"-1"}
        
        print_warning "WARNING: Downgrading database"
        print_info "Current revision:"
        alembic current
        
        read -p "Are you sure you want to downgrade? (yes/no): " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            print_info "Downgrade cancelled"
            exit 0
        fi
        
        alembic downgrade $TARGET
        
        if [ $? -eq 0 ]; then
            print_success "Database downgraded successfully!"
            print_info "New revision:"
            alembic current
        fi
        ;;
    
    "current")
        # Show current revision
        print_info "Current database revision:"
        alembic current -v
        ;;
    
    "history")
        # Show migration history
        print_info "Migration history:"
        alembic history -v
        ;;
    
    "heads")
        # Show all head revisions
        print_info "Head revisions:"
        alembic heads -v
        ;;
    
    "check")
        # Check if there are pending migrations
        print_info "Checking for pending migrations..."
        
        CURRENT=$(alembic current 2>/dev/null | grep -o '[a-f0-9]\{12\}' | head -1)
        HEAD=$(alembic heads 2>/dev/null | grep -o '[a-f0-9]\{12\}' | head -1)
        
        if [ "$CURRENT" = "$HEAD" ]; then
            print_success "Database is up to date!"
        else
            print_warning "There are pending migrations"
            print_info "Current: $CURRENT"
            print_info "Latest: $HEAD"
            print_info "Run '$0 upgrade' to apply pending migrations"
        fi
        ;;
    
    "init")
        # Initialize alembic (already done, but kept for completeness)
        print_warning "Alembic is already initialized"
        print_info "Use '$0 create \"message\"' to create a new migration"
        ;;
    
    "test")
        # Test upgrade and downgrade
        print_info "Testing migration (upgrade and downgrade)..."
        
        # Get current revision
        CURRENT=$(alembic current 2>/dev/null | grep -o '[a-f0-9]\{12\}' | head -1)
        
        # Upgrade to head
        print_info "Testing upgrade to head..."
        alembic upgrade head
        
        if [ $? -ne 0 ]; then
            print_error "Upgrade failed!"
            exit 1
        fi
        
        # Downgrade back
        print_info "Testing downgrade to original revision..."
        if [ -n "$CURRENT" ]; then
            alembic downgrade $CURRENT
        else
            alembic downgrade base
        fi
        
        if [ $? -eq 0 ]; then
            print_success "Migration test passed!"
        else
            print_error "Downgrade failed!"
            exit 1
        fi
        ;;
    
    "sql")
        # Show SQL for migration
        TARGET=${2:-"head"}
        print_info "SQL for migration to $TARGET:"
        alembic upgrade $TARGET --sql
        ;;
    
    "help"|*)
        # Show help
        print_info "SSA Database Migration Tool"
        print_info "=========================="
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  create <message>    Create a new migration"
        echo "  upgrade [target]    Upgrade database (default: head)"
        echo "  downgrade [target]  Downgrade database (default: -1)"
        echo "  current            Show current revision"
        echo "  history            Show migration history"
        echo "  heads              Show head revisions"
        echo "  check              Check for pending migrations"
        echo "  test               Test upgrade and downgrade"
        echo "  sql [target]       Show SQL for migration"
        echo "  help               Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 create \"Add user preferences table\""
        echo "  $0 upgrade"
        echo "  $0 downgrade -1"
        echo "  $0 check"
        ;;
esac

# Return to original directory
cd - > /dev/null