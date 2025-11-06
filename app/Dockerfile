
FROM node:20-alpine

WORKDIR /app

COPY package.json ./
# COPY package-lock.json ./

RUN npm install

COPY . .

RUN npx prisma generate --schema=src/prisma/schema.prisma
ENV NODE_ENV=development
ENV PORT=3000

EXPOSE 3000

CMD ["npm", "run", "dev"]
