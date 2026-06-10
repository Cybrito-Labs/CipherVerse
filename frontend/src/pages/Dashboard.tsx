import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Shield, Lock, FileKey, Zap, ArrowRight, ArrowUpRight } from 'lucide-react';

const popularTools = [
  { name: 'AES Encryption', path: '/symmetric/aes', icon: Lock, desc: 'Advanced standard for file and API encryption' },
  { name: 'RSA Encryption', path: '/asymmetric/rsa', icon: FileKey, desc: 'Public key system for secure data transmission' },
  { name: 'SHA-256 Hash', path: '/hashing/sha256', icon: Zap, desc: 'Generate cryptographic fingerprints for integrity' }
];

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
        <div className="flex items-center gap-2 text-[#EDEDED] mb-2">
          <Shield className="w-6 h-6" />
          <h1 className="text-3xl font-semibold tracking-tight">CipherVerse Workspace</h1>
        </div>
        <p className="text-[#A1A1AA] text-[15px] max-w-2xl leading-relaxed">
          The ultimate cryptography and cybersecurity toolkit. Encrypt, decrypt, hash, and encode data using modern, secure standards designed for developers and security engineers.
        </p>
      </motion.div>

      {/* Grid: Popular Tools */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <h2 className="text-lg font-semibold text-[#EDEDED] mb-4">Popular Tools</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {popularTools.map((tool) => (
            <Link key={tool.path} to={tool.path} className="group block">
              <div className="bg-[#0A0A0A] border border-[#27272A] rounded-[14px] p-5 shadow-sm transition-all duration-200 hover:border-[#52525B] hover:shadow-md h-full flex flex-col gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#171717] border border-[#27272A] flex items-center justify-center shadow-inner group-hover:scale-105 transition-transform duration-200">
                  <tool.icon className="w-5 h-5 text-[#EDEDED]" />
                </div>
                <div className="flex-1">
                  <h3 className="text-[15px] font-semibold text-[#EDEDED] flex items-center justify-between">
                    {tool.name}
                    <ArrowUpRight className="w-4 h-4 text-[#52525B] group-hover:text-[#EDEDED] transition-colors" />
                  </h3>
                  <p className="text-sm text-[#A1A1AA] mt-1 line-clamp-2">{tool.desc}</p>
                </div>
              </div>
            </Link>
          ))}
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
        <div className="bg-[#0A0A0A] border border-[#27272A] rounded-[14px] p-6 shadow-sm">
          <div className="flex items-center justify-between mb-6 border-b border-[#27272A] pb-4">
            <h2 className="text-lg font-semibold text-[#EDEDED]">Algorithm Standards</h2>
            <Link to="/classical" className="text-sm text-[#A1A1AA] hover:text-[#EDEDED] flex items-center gap-1 transition-colors">
              View all <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-12 pt-1 font-mono text-[11px] text-[#A1A1AA] font-bold">AES-GCM</div>
              <div>
                <p className="text-[13px] text-[#EDEDED] font-medium">Recommended for high-speed symmetric encryption</p>
                <p className="text-[12px] text-[#52525B] mt-0.5">Authenticated encryption providing both confidentiality and integrity.</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="w-12 pt-1 font-mono text-[11px] text-[#A1A1AA] font-bold">RSA-OAEP</div>
              <div>
                <p className="text-[13px] text-[#EDEDED] font-medium">Secure asymmetric padding</p>
                <p className="text-[12px] text-[#52525B] mt-0.5">Optimal Asymmetric Encryption Padding resists chosen-ciphertext attacks.</p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="w-12 pt-1 font-mono text-[11px] text-[#A1A1AA] font-bold">SHA-3</div>
              <div>
                <p className="text-[13px] text-[#EDEDED] font-medium">Modern hashing standard</p>
                <p className="text-[12px] text-[#52525B] mt-0.5">Based on Keccak sponge construction, highly resistant to collision.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity / Quick Actions */}
        <div className="bg-[#000000] border border-[#27272A] rounded-[14px] p-6 shadow-sm border-dashed flex flex-col items-center justify-center text-center gap-4 min-h-[300px]">
           <div className="w-12 h-12 rounded-full bg-[#171717] border border-[#27272A] flex items-center justify-center mb-2 shadow-inner">
             <Zap className="w-5 h-5 text-[#EDEDED]" />
           </div>
           <div>
             <h3 className="text-[#EDEDED] font-semibold">Quick Actions</h3>
             <p className="text-sm text-[#A1A1AA] mt-1 max-w-sm">Press <kbd className="font-mono text-[10px] bg-[#171717] px-1 py-0.5 rounded border border-[#27272A] text-[#EDEDED]">Cmd+K</kbd> anywhere to quickly search for algorithms, encodings, or certificates.</p>
           </div>
           <button className="mt-2 px-4 py-2 bg-[#EDEDED] text-[#000000] rounded-md font-medium text-sm hover:bg-[#D4D4D8] transition-colors shadow-sm">
             Open Command Palette
           </button>
        </div>
      </motion.div>
    </div>
  );
}
