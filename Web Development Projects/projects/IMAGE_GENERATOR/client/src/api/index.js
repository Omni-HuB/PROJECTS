// import axios from "axios";

// const API = axios.create({
//   baseURL: "https://image-generator-4l8k.onrender.com/api/",
// });

// export const GetPosts = async () => await API.get("/post/");
// export const CreatePost = async (data) => await API.post("/post/", data);
// export const GenerateAIImage = async (data) =>
//   await API.post("/generateImage/", data);


import axios from "axios";

const API = axios.create({
  baseURL: "https://image-generator-4l8k.onrender.com/api/",  // Ensure this matches backend routes
});

export const GetPosts = async () => {
  try {
    const response = await API.get("/post/");
    return response.data;
  } catch (error) {
    console.error("Error fetching posts:", error);
    throw error;
  }
};

export const CreatePost = async (data) => {
  try {
    const response = await API.post("/post/", data);
    return response.data;
  } catch (error) {
    console.error("Error creating post:", error);
    throw error;
  }
};

export const GenerateAIImage = async (data) => {
  try {
    const response = await API.post("/generateImage/", data);
    return response.data;
  } catch (error) {
    console.error("Error generating AI image:", error);
    throw error;
  }
};
