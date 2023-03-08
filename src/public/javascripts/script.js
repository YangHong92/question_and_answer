const form = document.querySelector('form');
const answer = document.querySelector('#answer');

const toggle_loading = (show) => {
  const wrapper = document.querySelector('#answer');
  const img = document.querySelector('#loading-icon');
  if(show){
    wrapper.style.filter = 'blur(3px)'
    img.style.display = 'block'
  } else {
    wrapper.style.filter = 'blur(0px)'
    img.style.display = 'none'
  }
}

const formEvent = form.addEventListener('submit', event => {
  event.preventDefault();
  const question = document.querySelector('#question');
  if (question.value) {
    toggle_loading(true)
    askQuestion(question.value);
  } else {
    answer.innerHTML = "You need to enter a question to get an answer.";
    answer.classList.add("error");
    }
});

const appendAnswer = (result) => {
  answer.innerHTML = `<p>${result.answer}</p>`;
  toggle_loading(false)
};

const askQuestion = (question) => {
  const params = {
    method: 'post',
    url: '/answer',
    headers: {
    'content-type': 'application/json'
  },
  data: { question }
  };
  axios(params)
    .then(response => {
    const answer = response.data;
    appendAnswer(answer);
    })
    .catch(error => {
      console.error(error)
    });
};