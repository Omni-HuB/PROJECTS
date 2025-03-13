// // import * as dotenv from "dotenv";
// // // import { Configuration, OpenAIApi } from "openai";
// // import { createError } from "../error.js";

// // import OpenAI from "openai";




// // dotenv.config();

// // // // Setup open ai api key
// // // const configuration = new Configuration({
// // //   apiKey: process.env.OPENAI_API_KEY,
// // // });

// // // const openai = new OpenAIApi(configuration);

// // const openai = new OpenAI(process.env.OPENAI_API_KEY);

// // // Controller to generate Image

// // export const generateImage = async (req, res, next) => {
// //   try {
// //     const { prompt } = req.body;

// //     // const response = await openai.createImage({
// //     //   prompt,
// //     //   n: 1,
// //     //   size: "1024x1024",
// //     //   response_format: "b64_json",
// //     // });

// //     const response = await openai.images.generate({
// //       model: "dall-e-3",
// //       prompt,
// //       n: 1,
// //       size: "1024x1024",
// //       response_format: "b64_json",
    
// //     });

// //     const generatedImage = response.data[0].b64_json;
// //     console.log(image.data[0].url);

// //     return res.status(200).json({ photo: generatedImage });
// //   } catch (error) {
// //     next(
// //       createError(
// //         error.status,
// //         error?.response?.data?.error?.message || error?.message
// //       )
// //     );
// //   }
// // };





// // // // import * as dotenv from "dotenv";
// // // // import OpenAI from "openai";  // ✅ Import default module
// // // // import { createError } from "../error.js";

// // // // dotenv.config();

// // // // // ✅ Correct way to initialize OpenAI
// // // // const openai = new OpenAI({
// // // //   apiKey: process.env.OPENAI_API_KEY,
// // // //   dangerouslyAllowBrowser: true,  // Add this if running in a browser environment
// // // // });

// // // // // export const generateImage = async (req, res, next) => {
// // // // //   try {
// // // // //     const { prompt } = req.body;

// // // // //     const response = await openai.images.generate({
// // // // //       model: "dall-e-3",
// // // // //       prompt: "a white siamese cat with blue eyes",
// // // // //       n: 1,
// // // // //       size: "1024x1024",
// // // // //       response_format: "b64_json",
// // // // //     });

// // // // //     const generatedImage = response.data[0].b64_json;  // Correct response structure

// // // // //     console.log(generatedImage);  // Debugging

// // // // //     return res.status(200).json({ photo: generatedImage });
// // // // //   } catch (error) {
// // // // //     next(
// // // // //       createError(
// // // // //         error.status || 500,
// // // // //         error?.response?.data?.error?.message || error?.message
// // // // //       )
// // // // //     );
// // // // //   }
// // // // // };


// // // import OpenAI from "openai";
// // // const openai = new OpenAI();

// // // const image = await openai.images.generate({ prompt: "A cute baby sea otter" });



// import * as dotenv from "dotenv";
// import OpenAI from "openai";
// import { createError } from "../error.js";

// dotenv.config();

// console.log("OpenAI API Key:", process.env.OPENAI_API_KEY);

// // Correct OpenAI Initialization
// const openai = new OpenAI({
//   apiKey: process.env.OPENAI_API_KEY,
// });

// export const generateImage = async (req, res, next) => {
//   try {
//     const { prompt } = req.body;

//     const response = await openai.images.generate({
//       model: "dall-e-3",
//       prompt,
//       n: 1,
//       size: "1024x1024",
//       response_format: "b64_json",
//     });

//     if (!response || !response.data) {
//       throw new Error("Failed to generate image");
//     }

//     // Correctly extract the base64 image
//     const generatedImage = response.data[0].b64_json;
    
//     console.log("Generated Image:", generatedImage); // Debugging

//     return res.status(200).json({ photo: generatedImage });
//   } catch (error) {
//     console.error("Error generating image:", error); // Debugging

//     next(
//       createError(
//         error.status || 500,
//         error?.response?.data?.error?.message || error?.message
//       )
//     );
//   }
// };


import * as dotenv from "dotenv";
import OpenAI from "openai";
import { createError } from "../error.js";

dotenv.config();

// Check if API Key is loading
console.log("OpenAI API Key:", process.env.OPENAI_API_KEY);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const generateImage = async (req, res, next) => {
  try {
    const { prompt } = req.body;
    if (!prompt) {
      throw new Error("Prompt is required");
    }

    const response = await openai.images.generate({
      model: "dall-e-3",
      prompt,
      n: 1,
      size: "1024x1024",
      response_format: "b64_json",
    });

    console.log("OpenAI Response:", response);

    const generatedImage = response?.data?.[0]?.b64_json;
    if (!generatedImage) throw new Error("Image generation failed");

    console.log("Generated Image:", generatedImage);

    return res.status(200).json({ photo: generatedImage });
  } catch (error) {
    console.error("Error generating image:", error.response?.data || error.message);

    next(
      createError(
        error.response?.status || 500,
        error.response?.data?.error?.message || "Failed to generate image"
      )
    );
  }
};
