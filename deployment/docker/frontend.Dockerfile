FROM node:18-alpine

WORKDIR /app/frontend

# Copy package.json and package-lock.json
COPY ./frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend code
COPY ./frontend .

# Set environment variables
ENV NODE_ENV=development

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]