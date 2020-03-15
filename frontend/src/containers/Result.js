import React from "react";
import "./Container.css";
import { connect } from "react-redux";
import { api_request } from "../store/actions/user";
import ArticleRecommender from "./resultcomponents/ArticleRecommender";
import SentimentAnalysis from "./resultcomponents/SentimentAnalysis";
import TopicDetection from "./resultcomponents/TopicDetection";
import LanguageDetection from "./resultcomponents/LanguageDetection";
import TextSummarization from "./resultcomponents/TextSummarization";
import EntityExtraction from "./resultcomponents/EntityExtraction";
import KeywordExtraction from "./resultcomponents/KeywordExtraction";
import ArticleExtraction from "./resultcomponents/ArticleExtraction";
import Lemmatization from "./resultcomponents/Lemmatization";
import Tokenize from "./resultcomponents/Tokenize";
import ConvertUppercaseToLower from "./resultcomponents/ConvertUppercaseToLower";
import RemoveNumbers from "./resultcomponents/RemoveNumbers";
import RemovePunctuation from "./resultcomponents/RemovePunctuation";
import RemoveWhiteSpaces from "./resultcomponents/RemoveWhiteSpaces";
import RemoveStopwords from "./resultcomponents/RemoveStopwords";
import Plagarism from "./resultcomponents/Plagarism";

class Result extends React.Component {

    getComponent = () => {
        switch (this.props.category) {
            case "Article Recommender":
                return <ArticleRecommender />;
            case "Sentiment Analysis":
                return <SentimentAnalysis />;
            case "Topic Detection":
                return <TopicDetection />;
            case "Language Detection":
                return <LanguageDetection />;
            case "Text Summarization":
                return <TextSummarization />;
            case "Entity Extraction":
                return <EntityExtraction />;
            case "Keyword Extraction":
                return <KeywordExtraction />;
            case "Article Extraction":
                return <ArticleExtraction />;
            case "Lemmatization":
                return <Lemmatization />;
            case "Tokenize":
                return <Tokenize />;
            case "Convert Uppercase To Lower":
                return <ConvertUppercaseToLower />;
            case "Remove Numbers":
                return <RemoveNumbers />;
            case "Remove Punctuation":
                return <RemovePunctuation />;
            case "Remove White Spaces":
                return <RemoveWhiteSpaces />;
            case "Remove Stopwords":
                return <RemoveStopwords />;
            case "Plagarism":
                return <Plagarism />;
            default:
                return <ArticleRecommender />;
        }
    }

    render() {
        return (
            <div>
                <div style={{ height: 30 }}></div>
                <div className="resultinput">
                    <div className="resultform">
                        <div><h5 className="title">Result</h5></div>
                        {this.getComponent()}
                    </div>
                </div>
            </div>
        )
    }
}

const mapStateToProps = state => {
    return {
        category: state.user.category
    };
};

export default connect(
    mapStateToProps, null
)(Result);