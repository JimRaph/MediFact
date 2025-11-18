
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { prisma } from '@/lib/db';

const PROTECTED_PATHS = ['/dashboard']; 

export async function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;

  if (!PROTECTED_PATHS.some(path => pathname.startsWith(path))) {
    return NextResponse.next();
  }

  const token = await getToken({ req: request });

  if (!token || !token.id) {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  const userId = token.id as string;
  
  const userExists = await prisma.user.findUnique({
    where: { id: userId },
    select: { id: true },
  });

  if (!userExists) {
    console.error(`Middleware Guard: DB record missing for User ID ${userId}. Forcing logout.`);

    const response = NextResponse.redirect(new URL('/login', request.url));


    response.cookies.delete('next-auth.session-token');
    response.cookies.delete('__Secure-next-auth.session-token');
    response.cookies.delete('session-token');

    return response;
  }

  return NextResponse.next();
}

export const config = {
    runtime: 'nodejs',
  matcher: ['/dashboard'],
};