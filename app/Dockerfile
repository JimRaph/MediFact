FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# Build-time generation (optional but ok)
RUN npx prisma generate --schema=src/prisma/schema.prisma

ENV NODE_ENV=development
ENV PORT=3000

EXPOSE 3000

# Runtime generation (important in dev)
CMD ["sh", "-c", "npx prisma generate --schema=src/prisma/schema.prisma && npm run dev"]
