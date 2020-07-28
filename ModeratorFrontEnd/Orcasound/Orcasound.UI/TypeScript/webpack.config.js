const path = require("path");

module.exports = {
  mode: 'development',
  entry: './src/index.ts',
  module: {
    rules: [
      {
        test: /\.(js)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader"
        }
      },
      {
        test: /\.(t|j)sx?$/,
        exclude: /node_modules/,
        use: {
          loader: 'awesome-typescript-loader'
        }
      }
    ],
  },
  resolve: {extensions: ['.js', '.jsx', '.ts', '.tsx']},
  output: {
    path: path.resolve(__dirname, '../wwwroot/js'),
    filename: "bundle.js",
    library: "ReactComponents",
    libraryTarget: "window"
  }
};