# 1) 建置：npm 安裝並以 Vite 打包成靜態檔
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --no-fund --no-audit
COPY . .
# 瀏覽器（在主機上）連線到後端的位址
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npx tsc -p tsconfig.app.json --noEmit && npx vite build

# 2) 執行：nginx 提供靜態檔
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
