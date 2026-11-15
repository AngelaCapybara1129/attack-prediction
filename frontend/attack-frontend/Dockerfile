# 1) 建置：pnpm 安裝並以 Vite 打包成靜態檔
FROM node:22-alpine AS build
WORKDIR /app
# 與本機相同的 pnpm 9（lockfileVersion 9.0）
RUN npm install -g pnpm@9 --no-fund --no-audit
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
# 瀏覽器（在主機上）連線到後端的位址
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=$VITE_API_URL
RUN pnpm exec tsc -p tsconfig.app.json --noEmit && pnpm exec vite build

# 2) 執行：nginx 提供靜態檔
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
