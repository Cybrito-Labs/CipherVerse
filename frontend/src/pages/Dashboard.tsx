import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Shield, Lock, FileKey, Zap, ArrowRight, ArrowUpRight, BookOpen, Sparkles } from 'lucide-react';
import { navigationGroups } from '@/constants/navigation';

const popularTools = [
  { name: 'AES Encryption', path: '/symmetric/aes', icon: Lock, desc: 'Advanced standard for file and API encryption' },
  { name: 'RSA Encryption', path: '/asymmetric/rsa', icon: FileKey, desc: 'Public key system for secure data transmission' },
  { name: 'SHA-256 Hash', path: '/hashing', icon: Zap, desc: 'Generate cryptographic fingerprints for integrity' }
];

// Extract all category hub suites (excluding root dashboard)
const categorySuites = navigationGroups.flatMap((group) =>
  group.items.filter((item) => item.path !== '/')
);

export default function DashboardPage() {
  return (
    <div className="max-w-[1200px] mx-auto space-y-10 pb-12">
      {/* Header Area */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="flex flex-col gap-2"
      >
        <div className="flex items-center gap-2 text-foreground mb-2">
          <Shield className="w-6 h-6" />
          <h1 className="text-3xl font-semibold tracking-tight">CipherVerse Workspace</h1>
        </div>
        <p className="text-muted-foreground text-[15px] max-w-2xl leading-relaxed">
          The ultimate cryptography and cybersecurity toolkit. Encrypt, decrypt, hash, and encode data using modern, secure standards designed for developers and security engineers.
        </p>
      </motion.div>

      {/* Grid: Popular Tools */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <h2 className="text-lg font-semibold text-foreground mb-4">Popular Tools</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {popularTools.map((tool) => (
            <Link key={tool.path} to={tool.path} className="group block">
              <div className="bg-card border border-border rounded-[14px] p-5 shadow-sm transition-all duration-200 hover:border-muted-foreground hover:shadow-md h-full flex flex-col gap-3">
                <div className="w-10 h-10 rounded-lg bg-secondary border border-border flex items-center justify-center shadow-inner group-hover:scale-105 transition-transform duration-200">
                  <tool.icon className="w-5 h-5 text-foreground" />
                </div>
                <div className="flex-1">
                  <h3 className="text-[15px] font-semibold text-foreground flex items-center justify-between">
                    {tool.name}
                    <ArrowUpRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{tool.desc}</p>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </motion.div>

      {/* Grid: All Security Suites & Categories (Search Engine Discovery Hub) */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Security Suites & Categories</h2>
            <p className="text-xs text-muted-foreground mt-0.5">Explore our comprehensive directories covering all cryptographic domains</p>
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3.5">
          {categorySuites.map((suite) => (
            <Link
              key={suite.path}
              to={suite.path}
              className="group p-4 rounded-xl bg-card border border-border hover:border-muted-foreground transition-all duration-200 shadow-sm flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="w-9 h-9 rounded-lg bg-secondary border border-border flex items-center justify-center group-hover:scale-105 transition-transform duration-200">
                    <suite.icon className="w-4 h-4 text-foreground" />
                  </div>
                  {suite.toolCount && (
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-secondary border border-border text-muted-foreground">
                      {suite.toolCount} tools
                    </span>
                  )}
                </div>
                <h3 className="text-sm font-semibold text-foreground group-hover:text-white transition-colors flex items-center justify-between">
                  {suite.label}
                  <ArrowRight className="w-3.5 h-3.5 opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-muted-foreground" />
                </h3>
                <p className="text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
                  {suite.description}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </motion.div>

      {/* Featured Cryptography Academy Spotlight Banner */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.15 }}
        className="p-6 sm:p-7 rounded-2xl border border-amber-500/30 bg-gradient-to-r from-amber-500/10 via-card to-background shadow-md relative overflow-hidden group"
      >
        <div className="absolute top-0 right-0 w-64 h-64 bg-amber-500/10 rounded-full blur-3xl pointer-events-none group-hover:scale-125 transition-transform duration-500" />
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>CipherVerse Academy • Start with Lesson 1</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-foreground group-hover:text-amber-200 transition-colors">
              New to Cryptography? Start Here: The Caesar Cipher &amp; ROT13 Guide
            </h2>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Master the foundational substitution cipher that started modern cryptanalysis. Explore historical Roman military origins under Julius Caesar, modular arithmetic formulas in ℤ₂₆, and automated frequency cracking.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 flex-shrink-0">
            <Link
              to="/blog/caesar-cipher"
              className="px-4 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-xs sm:text-sm shadow-md hover:opacity-90 transition-opacity inline-flex items-center gap-2"
            >
              <BookOpen className="w-4 h-4" />
              <span>Read Full Educational Guide</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/classical/caesar"
              className="px-4 py-2.5 rounded-xl border border-border bg-card/80 hover:bg-secondary text-foreground font-medium text-xs sm:text-sm transition-colors"
            >
              <span>Launch Interactive Solver</span>
            </Link>
          </div>
        </div>
      </motion.div>

      {/* Grid: Comparisons & Resources */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Security Standards Comparison */}
        <div className="bg-card border border-border rounded-[14px] p-6 shadow-sm">
          <div className="flex items-center justify-between mb-6 border-b border-border pb-4">
            <h2 className="text-lg font-semibold text-foreground">Algorithm Standards</h2>
            <Link to="/symmetric" className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors">
              Explore Suite <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-12 pt-1 font-mono text-[11px] text-muted-foreground font-bold">AES-GCM</div>
              <div>
                <p className="text-[13px] text-foreground font-medium">Recommended for high-speed symmetric encryption</p>
                <p className="text-[12px] text-muted-foreground mt-0.5">Authenticated encryption providing both confidentiality and integrity.</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="w-12 pt-1 font-mono text-[11px] text-muted-foreground font-bold">RSA-OAEP</div>
              <div>
                <p className="text-[13px] text-foreground font-medium">Secure asymmetric padding</p>
                <p className="text-[12px] text-muted-foreground mt-0.5">Optimal Asymmetric Encryption Padding resists chosen-ciphertext attacks.</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="w-12 pt-1 font-mono text-[11px] text-muted-foreground font-bold">SHA-3</div>
              <div>
                <p className="text-[13px] text-foreground font-medium">Modern hashing standard</p>
                <p className="text-[12px] text-muted-foreground mt-0.5">Based on Keccak sponge construction, highly resistant to collision.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity / Quick Actions */}
        <div className="bg-background border border-border rounded-[14px] p-6 shadow-sm border-dashed flex flex-col items-center justify-center text-center gap-4 min-h-[300px]">
           <div className="w-12 h-12 rounded-full bg-secondary border border-border flex items-center justify-center mb-2 shadow-inner">
             <Zap className="w-5 h-5 text-foreground" />
           </div>
           <div>
             <h3 className="text-foreground font-semibold">Quick Actions</h3>
             <p className="text-sm text-muted-foreground mt-1 max-w-sm">Press <kbd className="font-mono text-[10px] bg-secondary px-1 py-0.5 rounded border border-border text-foreground">Cmd+K</kbd> anywhere to quickly search for algorithms, encodings, or certificates.</p>
           </div>
           <button className="mt-2 px-4 py-2 bg-foreground text-background rounded-md font-medium text-sm hover:bg-[#D4D4D8] transition-colors shadow-sm">
             Open Command Palette
           </button>
        </div>
      </motion.div>
    </div>
  );
}
