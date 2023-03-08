const axios = require('axios');
const express = require('express');
const Filter = require('bad-words');
const router = express.Router();

const apiKey = process.env['OPENAI_API_KEY'];
const client = axios.create({
  headers: { 'Authorization': 'Bearer ' + apiKey }
});

const endpoint = 'http://127.0.0.1:8000/ask'

router.post('/', (req, res) => {
  // call the backend Q&A API
  // let filter = new Filter();
  // if (filter.isProfane(req.body.question)) {
  //   res.send({ "answer": "That’s not a question we can answer." });
  //   return;
  // }
  console.log("send data: ", req.body.question)
  client.post(endpoint, {query: req.body.question})
    .then(result => {
      res.send({ "answer": result.data.answer })
    }).catch(result => {
      res.send({ "answer": "Sorry, I don't have an answer." })
    });
});

module.exports = router;