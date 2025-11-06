import {prisma} from "@/lib/db"
import bcrypt from "bcryptjs" 
import { NextResponse } from "next/server"

export async function POST(req: Request){
    try {
        const {name, email, password} = await req.json()

        if (!email || !password){
            return NextResponse.json({
                error: "Missing fields"
            },
        {status: 400})
        }

        const existingUser = await prisma.user.findUnique({where: {email}})
        if(existingUser){
            return NextResponse.json({error: "Email already registered"},
                {status: 400}
            )
        }

        const hashedPassword = await bcrypt.hash(password, 10)

        const user = await prisma.user.create({
            data: {
                name,
                email,
                hashedPassword,
            }
        })

        return NextResponse.json(
            { message: "User registered successfully", user: { id: user.id, email: user.email } },
            { status: 201 }
        )
    } catch (error) {
        console.error("Registration error: ", error)
        return NextResponse.json({
            error: "Something went wrong"
        }, {status: 500})
    }
}