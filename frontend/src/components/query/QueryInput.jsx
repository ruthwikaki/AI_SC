import React, { useState, useRef, useEffect } from 'react';
import { FaSearch, FaMicrophone, FaSpinner, FaChevronDown } from 'react-icons/fa';

const QueryInput = ({ onSubmitQuery, isLoading, modelConfig, setModelConfig }) => {
  const [query, setQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [showModelOptions, setShowModelOptions] = useState(false);
  const inputRef = useRef(null);
  const modelOptionsRef = useRef(null);
  
  // Focus input on component mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
    
    // Add click outside listener for model options dropdown
    const handleClickOutside = (event) => {
      if (modelOptionsRef.current && !modelOptionsRef.current.contains(event.target)) {
        setShowModelOptions(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmitQuery(query);
    }
  };
  
  // Handle microphone recording
  const handleVoiceInput = () => {
    // Check if browser supports speech recognition
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Voice input is not supported in your browser.');
      return;
    }
    
    // Toggle recording state
    setIsRecording(!isRecording);
    
    if (!isRecording) {
      // Start recording
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuery(transcript);
        setIsRecording(false);
      };
      
      recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        setIsRecording(false);
      };
      
      recognition.onend = () => {
        setIsRecording(false);
      };
      
      recognition.start();
    }
  };
  
  // Available models
  const availableModels = [
    { id: 'mistral-7b-int8-gptq', name: 'Mistral 7B (Int8, GPTQ)', speed: 'Fast' },
    { id: 'mistral-7b-int4-gptq', name: 'Mistral 7B (Int4, GPTQ)', speed: 'Fastest' },
    { id: 'llama3-8b-int8-gptq', name: 'LLaMA3 8B (Int8, GPTQ)', speed: 'Medium' },
    { id: 'supply-chain-finetuned-7b', name: 'Supply Chain Fine-tuned (FP16)', speed: 'Slow' },
  ];
  
  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        {/* Model selection dropdown */}
        <div className="absolute left-3 top-3 z-10">
          <div className="relative" ref={modelOptionsRef}>
            <button
              type="button"
              onClick={() => setShowModelOptions(!showModelOptions)}
              className="flex items-center px-3 py-1 text-sm text-gray-500 bg-gray-100 rounded-md hover:bg-gray-200 focus:outline-none"
            >
              <span className="truncate max-w-[120px] md:max-w-xs">
                {availableModels.find(m => m.id === modelConfig.model)?.name || modelConfig.model}
              </span>
              <FaChevronDown className="ml-1 h-3 w-3" />
            </button>
            
            {showModelOptions && (
              <div className="absolute top-full left-0 mt-1 w-64 bg-white rounded-md shadow-lg z-50 py-1 border border-gray-200">
                {availableModels.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => {
                      setModelConfig({...modelConfig, model: model.id});
                      setShowModelOptions(false);
                    }}
                    className={`
                      w-full text-left px-4 py-2 text-sm hover:bg-gray-100 flex items-center justify-between
                      ${modelConfig.model === model.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'}
                    `}
                  >
                    <span>{model.name}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      model.speed === 'Fast' ? 'bg-green-100 text-green-800' :
                      model.speed === 'Fastest' ? 'bg-blue-100 text-blue-800' :
                      model.speed === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-orange-100 text-orange-800'
                    }`}>
                      {model.speed}
                    </span>
                  </button>
                ))}
                
                <div className="border-t border-gray-200 mt-1 pt-1 px-4 py-2">
                  <div className="flex items-center justify-between text-sm">
                    <label htmlFor="temperature" className="text-gray-700">Temperature:</label>
                    <span className="text-blue-600 font-medium">{modelConfig.temperature}</span>
                  </div>
                  <input
                    id="temperature"
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={modelConfig.temperature}
                    onChange={(e) => setModelConfig({...modelConfig, temperature: parseFloat(e.target.value)})}
                    className="w-full mt-1"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Precise</span>
                    <span>Creative</span>
                  </div>
                </div>
                
                <div className="border-t border-gray-200 mt-1 pt-1 px-4 py-2">
                  <div className="flex items-center justify-between text-sm">
                    <label htmlFor="max_tokens" className="text-gray-700">Max Tokens:</label>
                    <span className="text-blue-600 font-medium">{modelConfig.max_tokens}</span>
                  </div>
                  <input
                    id="max_tokens"
                    type="range"
                    min="100"
                    max="2000"
                    step="100"
                    value={modelConfig.max_tokens}
                    onChange={(e) => setModelConfig({...modelConfig, max_tokens: parseInt(e.target.value)})}
                    className="w-full mt-1"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Short</span>
                    <span>Long</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Main query input */}
        <div className="relative">
          <textarea
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a supply chain question..."
            className="w-full py-3 pl-36 pr-24 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none overflow-hidden"
            style={{ minHeight: '60px', maxHeight: '120px' }}
            rows={query.split('\n').length > 3 ? 3 : query.split('\n').length || 1}
            disabled={isLoading}
          />
          
          <div className="absolute right-3 top-3 flex">
            {/* Voice input button */}
            <button
              type="button"
              onClick={handleVoiceInput}
              className={`mr-2 p-2 rounded-full focus:outline-none ${
                isRecording 
                  ? 'bg-red-100 text-red-600 animate-pulse' 
                  : 'text-gray-500 hover:bg-gray-100'
              }`}
              disabled={isLoading}
            >
              <FaMicrophone className="h-5 w-5" />
            </button>
            
            {/* Submit button */}
            <button
              type="submit"
              className={`
                p-2 rounded-full focus:outline-none
                ${isLoading 
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed' 
                  : query.trim() 
                    ? 'bg-blue-600 text-white hover:bg-blue-700' 
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                }
              `}
              disabled={isLoading || !query.trim()}
            >
              {isLoading ? (
                <FaSpinner className="h-5 w-5 animate-spin" />
              ) : (
                <FaSearch className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default QueryInput;