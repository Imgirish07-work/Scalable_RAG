# Multi-stage build: compile the SPA with node, ship only static files via nginx.
#
# Build args:
#   VITE_API_BASE_URL  Baked into the bundle at build time (Vite limitation).
#                      Defaults to `/api`, which routes through nginx's reverse
#                      proxy to the backend — kills cross-origin entirely.

FROM node:20-alpine AS builder

WORKDIR /app

# Install deps first so this layer caches across source edits.
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


FROM nginx:1.27-alpine AS runtime

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://localhost/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
