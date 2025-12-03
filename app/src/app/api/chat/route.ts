import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/db'
import { generateConversationTitle } from '@/lib/utils'
import { ChatRequest } from '@/types/chat'
import {  RagResponse } from '@/types/api'
import { getRagAnswer, healthCheck } from '@/lib/api';
import { rateLimitService } from '@/lib/rateLimiter' 
import { headers } from 'next/headers'


const getClientIp = async (): Promise<string> => {
  const headersList = await headers();
  const ipHeader =  headersList.get("x-forwarded-for")
  const ip = ipHeader ? ipHeader.split(",")[0].trim() : '127.0.0.1';
  return ip; 

}



export async function POST(request: NextRequest) {
  const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL;

  if (!RAG_SERVICE_URL) {
    console.error("Configuration Error: RAG_SERVICE_URL is not defined.");
    return NextResponse.json(
      { error: 'RAG service URL is not configured.' },
      { status: 500 }
    );
  }
  console.log('DEBUGG..........ING ')

  try {

    const session = await getServerSession(authOptions)
    const ip = await getClientIp();
    const userId = session?.user?.id 
  
    const limitStatus = await rateLimitService.acquireLock(ip, userId);

    if(!limitStatus.allowed){
      const headers = {
        "X-RateLimit-Limit": rateLimitService.MAX_REQUESTS.toString(),
        "X-RateLimit-Remaining": "0",
      }
      console.warn(`Rate limit exceeded for IP: ${ip}, User: ${userId || 'N/A'}`)
      return NextResponse.json(
      {error: "Rate limit exceeded. Please try again later."},
      {status: 429, headers}
      )
    }

    console.log(`[AUTH] session retrieved: ${session ? 'YES' : 'NO'}`);
    if (session && session.user){
      console.log(`[AUTH] user id: ${session.user.id || 'N/A'}`)
    }

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const isRagServiceHealthy = await healthCheck();

    console.log('rag urrl: ', RAG_SERVICE_URL)

    if (!isRagServiceHealthy) {
      console.log('rag url: ', RAG_SERVICE_URL)
      console.error('RAG Service is unavailable or failed health check.');
      return NextResponse.json(
        { error: 'RAG service is currently unavailable. Please try again later.' },
        { status: 503 }
      );
    }

    const body: ChatRequest = await request.json()
    const { message, conversationId, history } = body

    if (!message?.trim()) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    let conversation

 
    if (conversationId) {
      conversation = await prisma.conversation.findFirst({
        where: {
          id: conversationId,
          userId: session.user.id,
        },
      })

      if (!conversation) {
        return NextResponse.json(
          { error: 'Conversation not found' },
          { status: 404 }
        )
      }
    } else {

      conversation = await prisma.conversation.create({
        data: {
          title: generateConversationTitle(message),
          userId: session.user.id,
        },
      })
    }


    await prisma.message.create({
      data: {
        role: 'user',
        content: message,
        conversationId: conversation.id,
        source: ''
      },
    })

    const ragResponse: RagResponse = await getRagAnswer(message, history || []); 

    console.log("ragResponse: ", ragResponse)
    const aiAnswer = ragResponse.answer;
    const sources = ragResponse.sources;

    await prisma.message.create({
      data: {
        role: 'assistant',
        content: aiAnswer,
        conversationId: conversation.id,
        source: JSON.stringify(sources)
      },
    })

    await prisma.conversation.update({
      where: { id: conversation.id },
      data: { updatedAt: new Date() },
    })

    const successHeaders = {
      "X-RateLmit-Limit": rateLimitService.MAX_REQUESTS.toString(),
      "X-RateLimit-Remaining": limitStatus.ipRemaining.toString()
    }

    return NextResponse.json({
      response: aiAnswer,
      conversationId: conversation.id,
      sources: sources,
    }, {headers: successHeaders})
  } catch (error) {
    console.error('Chat API Error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}