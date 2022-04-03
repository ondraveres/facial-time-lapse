FROM node:14-alpine

WORKDIR /app

COPY package.json package.json
COPY package-lock.json package-lock.json

COPY . .

RUN echo runner $CMD_RUNNER
RUN npm install

EXPOSE 3000

CMD ["node", "./server"]
