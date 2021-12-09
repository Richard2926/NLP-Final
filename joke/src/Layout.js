import ModelSelect from './ModelSelect';
import React from 'react';
import gptjokes from './jokes/gpt';
import ngramjokes from './jokes/ngram';
import Rating from './Rating';

import { initializeApp } from 'firebase/app';
import { 
    getFirestore,
    getDoc,
    updateDoc,
    doc } from 'firebase/firestore/lite';


const firebaseConfig = {
  apiKey: "AIzaSyCMJIIbPFFgJ14qV1CL4Hz7HE65GWnc16Y",
  authDomain: "nlp-final-7dba7.firebaseapp.com",
  projectId: "nlp-final-7dba7",
  storageBucket: "nlp-final-7dba7.appspot.com",
  messagingSenderId: "971133946111",
  appId: "1:971133946111:web:85688dd44f0d3ecfaea4ca",
  measurementId: "G-E9D41DNWMZ"
};


const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

class Layout extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            "N-gram": {
                "jokes": ngramjokes,
                "index": Math.floor(Math.random() * ngramjokes.length)
            },
            "GPT-2": {
                "jokes": gptjokes,
                "index": Math.floor(Math.random() * gptjokes.length)
            },
            rating: 0,
            model: "N-gram",
            joke: "Pick a model and hit generate to create a joke using that model ! Then, rate the model with the following scale below. Go ahead and try it !"
        }; 
        this.handleModelChange = this.handleModelChange.bind(this);
        this.generate = this.generate.bind(this);
        this.updateRating = this.updateRating.bind(this);
    }

    handleModelChange(model) {
        this.setState({
            model: model,
            joke: "Pick a model and hit generate to create a joke using that model ! Then, rate the model with the following scale below. Go ahead and try it !"
            ,rating: 0
        })
    }

    updateRating(value) {
        // console.log(value);
        this.setState({
            rating: value
        })
    }
    async generate() {
        if (this.state.joke != "Pick a model and hit generate to create a joke using that model ! Then, rate the model with the following scale below. Go ahead and try it !") {
            let ref = doc(db, ('jokes/' + this.state.model));
            console.log('jokes/' + this.state.model);
            let data = await getDoc(ref);
            console.log(data.data());
            let updatedJokes = data.data()["jokes"];
            let temp = updatedJokes[this.state.joke];
            console.log(this.state.joke);
            console.log(temp);
            temp[this.state.rating] = temp[this.state.rating] + 1;
            updatedJokes[this.state.joke] = temp;
            await updateDoc(ref, "jokes", updatedJokes);            
        }
        let curr = this.state[this.state.model];
        let index = Math.floor(Math.random() * curr["jokes"].length);
        console.log(index)
        console.log(curr["jokes"])
        
        this.setState({
            joke: curr["jokes"][index],
            rating: 0
        })
        console.log("updated");
    }

    render() {
      return (
          <div>
            <ModelSelect handleModelChange={this.handleModelChange}/>
            <p class="font-mono text-2xl mt-20 text-white">
                {this.state.joke}
            </p>
            <div class="h-48 flex flex-wrap content-start mt-8 items-center justify-center">
                <Rating updateRating={this.updateRating} value={0} rating={this.state.rating} text="Not Cohesive"/>
                <Rating updateRating={this.updateRating} value={1} rating={this.state.rating} text="Somewhat Cohesive"/>
                <Rating updateRating={this.updateRating} value={2} rating={this.state.rating} text="Somewhat Funny"/>
                <Rating updateRating={this.updateRating} value={3} rating={this.state.rating} text="Funny"/>
                <Rating updateRating={this.updateRating} value={4} rating={this.state.rating} text="Made Me Laugh"/>
            </div>
            <button class="bg-white hover:bg-black hover:text-white text-black italic py-2 px-4 rounded" onClick={this.generate}>
                Generate
            </button>
          </div>
      );
    }
}

export default Layout;