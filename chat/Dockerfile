FROM node:19 as builder-production

WORKDIR /app
EXPOSE 3000

RUN git clone https://github.com/huggingface/chat-ui.git \
        && cd chat-ui \
        && git checkout 75604494d22bb44694c23c34c637d53be224ac65
        # && git checkout 0ad340edea015777a2c5cb5fe68044ecff53b13f
        
        
RUN cp -R /app/chat-ui/*.json /app/

RUN sed -i 's/127.0.0.1:27017/mongodb:27017/g' /app/package.json

RUN --mount=type=cache,target=/app/.npm \
        npm set cache /app/.npm && \
        npm ci --omit=dev

FROM builder-production as builder

RUN --mount=type=cache,target=/app/.npm \
        npm set cache /app/.npm && \
        npm ci

RUN cp -R /app/chat-ui/* /app/

COPY .env.local /app/.env.local
COPY apply_patches.py /app/apply_patches.py
COPY patches.json /app/patches.json
RUN chown -R 1000 /app/* 

RUN apt-get update && apt-get install python -y
RUN python /app/apply_patches.py

RUN --mount=type=secret,id=DOTENV_LOCAL,dst=.env.local \
    npm run build

FROM node:19-slim

RUN npm install -g pm2

COPY --from=builder-production /app/node_modules /app/node_modules
COPY --from=builder /app/chat-ui/package.json /app/package.json
COPY --from=builder /app/build /app/build

RUN apt-get -y update; apt-get -y install curl

WORKDIR /app/build
CMD pm2 start /app/build/index.js -i 1 --no-daemon
