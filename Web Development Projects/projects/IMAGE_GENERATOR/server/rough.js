
import * as dotenv from "dotenv";
import { Configuration, OpenAIApi } from "openai";
import { createError } from "../error.js";

console.log("OpenAI API Key:", process.env.OPENAI_API_KEY);