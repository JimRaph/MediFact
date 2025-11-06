import { prisma } from '@/lib/db'
import dotenv from 'dotenv'
dotenv.config()

async function testConnection() {
  try {
    console.log('Testing database connection...')
    
    await prisma.$queryRaw`SELECT 1 as connected`
    console.log('Database connection successful!')
    
    const userCount = await prisma.user.count()
    console.log(`Users in database: ${userCount}`)
    
  } catch (error) {
    console.error('Database connection failed:', error)
  } finally {
    await prisma.$disconnect()
  }
}

testConnection()