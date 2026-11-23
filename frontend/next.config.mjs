/** @type {import('next').NextConfig} */
// "standalone" lets the Docker image ship only the files the server needs.
export default { reactStrictMode: true, output: "standalone" };
