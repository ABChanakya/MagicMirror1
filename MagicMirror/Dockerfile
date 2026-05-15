FROM arm64v8/node:22-slim

# git is required by MagicMirror's postinstall script (git clean -df fonts vendor)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/magicmirror

COPY package.json package-lock.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY . .

EXPOSE 8080 8082

ENV MM_CONFIG_FILE=config/config.child.js \
    NODE_ENV=production

CMD ["node", "serveronly"]
