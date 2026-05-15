import { useState, useRef, useEffect } from 'react';
import { Send, BookOpen, Loader2, Bot, User, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

interface Source {
  title: string;
  trigger_snippet: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: '你好！我是中国哲学史研究助手。你可以询问关于《老子》、《庄子》、儒家经典等哲学文本的问题。我会基于已有的哲学史著作为你提供严谨的学术性回答。',
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 接收到的数据清洗函数：删除 JSON 数组和引用来源文本
  const cleanData = (text: string) => {
    // 1. 删除 JSON 数组（从 [ 开始到 ] 结束）
    let cleaned = text.replace(/\[\{.*?\}\]/gs, '');

    // 2. 删除"引用来源"及之后的内容（包括各种变体）
    cleaned = cleaned.replace(/引用来源[：:].*$/s, '');
    cleaned = cleaned.replace(/【章节\d+.*$/s, '');
    // 3. 删除独立的章节标记行（如单独一行的"【章节2：三 性善】"）
    cleaned = cleaned.replace(/\n【章节\d+.*$/gm, '');

    return cleaned.trim();
  };
  // 发送消息（流式）
  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInput('');
    setIsLoading(true);

    try {
      abortControllerRef.current = new AbortController();

      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content, stream: true }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error('网络请求失败');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('无法读取响应');

      let accumulatedContent = '';
      let sources: Source[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            // 关键修改：累加数据并立即清洗
            accumulatedContent += data;
            const cleanedContent = cleanData(accumulatedContent);

            setMessages(prev =>
              prev.map(msg =>
                msg.id === assistantMessage.id
                  ? { ...msg, content: cleanedContent }
                  : msg
              )
            );
          } else if (line.startsWith('event: sources')) {
            // 下一行是来源数据
            const nextLine = lines[lines.indexOf(line) + 1];
            if (nextLine?.startsWith('data: ')) {
              try {
                sources = JSON.parse(nextLine.slice(6));
              } catch (e) {
                console.error('解析来源失败:', e);
              }
            }
          }
        }
      }

      // 最终更新：确保最后一次清洗并添加来源
      const finalCleanedContent = cleanData(accumulatedContent);
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? { ...msg, content: finalCleanedContent, sources, isStreaming: false }
            : msg
        )
      );

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('用户取消请求');
      } else {
        console.error('请求错误:', error);
        setMessages(prev =>
          prev.map(msg =>
            msg.id === assistantMessage.id
              ? { ...msg, content: '抱歉，请求出现错误。请稍后重试。', isStreaming: false }
              : msg
          )
        );
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const stopGeneration = () => {
    abortControllerRef.current?.abort();
    setIsLoading(false);
    setMessages(prev =>
      prev.map(msg =>
        msg.isStreaming ? { ...msg, isStreaming: false } : msg
      )
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col">
      {/* 头部 */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-2 rounded-lg">
            <BookOpen className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-800">哲学史问答系统</h1>
            <p className="text-xs text-slate-500">基于RAG的智能学术助手</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-600">
          <Sparkles className="w-4 h-4 text-amber-500" />
          <span>GLM-4-Flash</span>
        </div>
      </header>

      {/* 消息区域 */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              {/* 头像 */}
              <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${message.role === 'user' ? 'bg-indigo-100 text-indigo-600' : 'bg-emerald-100 text-emerald-600'
                }`}>
                {message.role === 'user' ? <User size={20} /> : <Bot size={20} />}
              </div>

              {/* 消息内容 */}
              <div className={`flex-1 max-w-[80%] ${message.role === 'user' ? 'items-end' : 'items-start'} flex flex-col`}>
                <div className={`rounded-2xl px-5 py-3 shadow-sm ${message.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-br-none'
                  : 'bg-white border border-slate-200 rounded-bl-none'
                  }`}>
                  <div className={`max-w-none ${message.role === 'user' ? 'text-white' : 'text-slate-800'
                    }`}>
                    {message.role === 'assistant' ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          // 段落样式：自然段间距
                          p: ({ children }) => <p className="mb-3 leading-relaxed last:mb-0">{children}</p>,
                          // 一级标题：大字号，粗体，上边距
                          h1: ({ children }) => <h1 className="text-xl font-bold mb-4 mt-6 text-slate-900">{children}</h1>,
                          // 二级标题：中等字号，粗体，上边距
                          h2: ({ children }) => <h2 className="text-lg font-bold mb-3 mt-5 text-slate-800">{children}</h2>,
                          // 三级标题：普通字号，粗体，颜色略深
                          h3: ({ children }) => <h3 className="text-base font-bold mb-2 mt-4 text-slate-800">{children}</h3>,
                          // 无序列表：圆点，缩进，间距
                          ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1 marker:text-slate-400">{children}</ul>,
                          // 有序列表：数字，缩进，间距
                          ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1 marker:text-slate-500">{children}</ol>,
                          // 列表项：行高舒适
                          li: ({ children }) => <li className="leading-relaxed pl-1">{children}</li>,
                          // 强调/粗体：使用主题色突出
                          strong: ({ children }) => <strong className="font-bold text-indigo-700">{children}</strong>,
                          // 引用块：左边框，背景色，斜体
                          blockquote: ({ children }) => (
                            <blockquote className="border-l-4 border-indigo-300 pl-4 py-1 my-3 bg-slate-50 italic text-slate-600 rounded-r">
                              {children}
                            </blockquote>
                          ),
                          // 代码：等宽字体，背景色
                          code: ({ children }) => (
                            <code className="bg-slate-100 px-1.5 py-0.5 rounded text-sm font-mono text-pink-600">
                              {children}
                            </code>
                          ),
                          // 链接：主题色，下划线悬停
                          a: ({ children, href }) => (
                            <a href={href} className="text-indigo-600 hover:underline hover:text-indigo-800 transition-colors">
                              {children}
                            </a>
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    ) : (
                      // 用户消息保持纯文本，但保留换行
                      <div className="whitespace-pre-wrap leading-relaxed">
                        {message.content}
                      </div>
                    )}
                    {message.isStreaming && (
                      <span className="inline-block w-2 h-4 bg-current ml-1 animate-pulse mt-2" />
                    )}
                  </div>
                </div>

                {/* 来源信息（仅助手消息） */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200 text-sm w-full">
                    <div className="flex items-center gap-2 text-slate-600 mb-2 font-medium">
                      <BookOpen size={14} />
                      <span>引用来源</span>
                    </div>
                    <div className="space-y-2">
                      {message.sources.map((source, idx) => (
                        <div key={idx} className="flex gap-2 text-xs">
                          <span className="text-indigo-600 font-medium min-w-[20px]">[{idx + 1}]</span>
                          <div className="flex-1">
                            <div className="font-medium text-slate-700">《{source.title}》</div>
                            <div className="text-slate-500 mt-0.5 line-clamp-2">
                              触发片段: {source.trigger_snippet}...
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* 输入区域 */}
      <footer className="bg-white border-t border-slate-200 px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="relative flex items-end gap-2 bg-slate-50 rounded-2xl border border-slate-200 p-2 focus-within:ring-2 focus-within:ring-indigo-500 focus-within:border-transparent transition-all">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的哲学问题...（Shift+Enter换行）"
              rows={1}
              className="flex-1 bg-transparent px-3 py-2 outline-none resize-none max-h-32 text-slate-700 placeholder:text-slate-400"
              style={{ minHeight: '44px' }}
              disabled={isLoading}
            />
            <button
              onClick={isLoading ? stopGeneration : sendMessage}
              className={`p-2 rounded-xl transition-all ${isLoading
                ? 'bg-red-100 text-red-600 hover:bg-red-200'
                : input.trim()
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md'
                  : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                }`}
              disabled={!isLoading && !input.trim()}
            >
              {isLoading ? (
                <div className="flex items-center gap-1">
                  <div className="w-4 h-4 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs font-medium px-1">停止</span>
                </div>
              ) : (
                <Send size={20} />
              )}
            </button>
          </div>
          <div className="text-center mt-2 text-xs text-slate-400">
            基于中国哲学史著作检索 · 生成内容仅供参考
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;