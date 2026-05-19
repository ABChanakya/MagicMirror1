FROM arm64v8/node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global user.email "mm@jetson.com" && \
    git config --global user.name "MagicMirror"

WORKDIR /opt/magicmirror

COPY package.json package-lock.json ./

RUN git init && git add -A && git commit -m "init" && \
    npm install --omit=dev --no-audit --no-fund

COPY . .

EXPOSE 8080 8081 8082

ENV MM_CONFIG_FILE=config/config.child.js \
    NODE_ENV=production

CMD ["node", "--max-old-space-size=512", "serveronly"]
