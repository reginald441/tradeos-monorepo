# =============================================================================
# TradeOS Frontend Dockerfile
# Multi-stage build: Node.js build + Nginx serve
# =============================================================================

# =============================================================================
# STAGE 1: Dependencies
# =============================================================================
FROM node:18-alpine AS dependencies

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* yarn.lock* ./

# Install dependencies based on available lock file
RUN if [ -f yarn.lock ]; then \
      yarn install --frozen-lockfile --production=false; \
    elif [ -f package-lock.json ]; then \
      npm ci --include=dev; \
    else \
      npm install --include=dev; \
    fi

# =============================================================================
# STAGE 2: Builder
# =============================================================================
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy dependencies from previous stage
COPY --from=dependencies /app/node_modules ./node_modules

# Copy source code
COPY . .

# Build arguments for environment variables
ARG REACT_APP_API_URL
ARG REACT_APP_WS_URL
ARG REACT_APP_VERSION
ARG NODE_ENV=production

# Set environment variables for build
ENV REACT_APP_API_URL=${REACT_APP_API_URL} \
    REACT_APP_WS_URL=${REACT_APP_WS_URL} \
    REACT_APP_VERSION=${REACT_APP_VERSION} \
    NODE_ENV=${NODE_ENV} \
    GENERATE_SOURCEMAP=false \
    CI=true

# Build the application
RUN if [ -f yarn.lock ]; then \
      yarn build; \
    else \
      npm run build; \
    fi

# =============================================================================
# STAGE 3: Development
# =============================================================================
FROM node:18-alpine AS development

WORKDIR /app

# Copy dependencies
COPY --from=dependencies /app/node_modules ./node_modules

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "start"]

# =============================================================================
# STAGE 4: Production (Nginx)
# =============================================================================
FROM nginx:alpine AS production

LABEL maintainer="TradeOS Team <dev@tradeos.io>" \
      version="1.0.0" \
      description="TradeOS Frontend Web Application"

# Install additional packages for health checks
RUN apk add --no-cache curl

# Remove default nginx configuration
RUN rm -rf /etc/nginx/conf.d/default.conf \
    /usr/share/nginx/html/*

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built application from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Create health check endpoint
RUN echo '<!DOCTYPE html><html><head><title>Health</title></head><body>OK</body></html>' > /usr/share/nginx/html/health

# Set proper permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Add security headers configuration
RUN echo 'server_tokens off;' > /etc/nginx/conf.d/security.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
