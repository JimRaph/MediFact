import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/db'



export async function GET(_request: NextRequest) {
  try {

    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const conversations = await prisma.conversation.findMany({
      where: {
        userId: session.user.id,
      },
      include: {
        messages: {
          orderBy: {
            createdAt: 'asc',
          },
          take: 1, 
        },
      },
      orderBy: {
        updatedAt: 'desc',
      },
    })

    return NextResponse.json(conversations)
  } catch (error) {
    console.error('Conversations API Error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// export async function DELETE(
//   _request: Request,
//   {params}: {params: {id: string}}
// ){
//   const session = await getServerSession(authOptions);

//   if(!session || !session.user || !session.user.id) {
//     return new NextResponse('Unathorized', {status: 401});
//   }

//   const conversationId = params.id 
//   console.log('what')

//   try{
//     await prisma.conversation.delete({
//       where: {
//         id: conversationId,
//         userId: session.user.id,
//       },
//     });

//     return new NextResponse('Conversation deleted', {status: 200});
//   } catch(error) {
//     console.error('Error deleting conversation: ', error)
//     if (typeof error === 'object' && error !== null && 'code' in error && typeof error.code === 'string'){
//       if (error.code === 'P2025') {
//     return new NextResponse('Conversation not found or unauthorized', { status: 404 });
//   }
//     }
//     return new NextResponse('Server error, try again', {status: 500})
//   }
// }