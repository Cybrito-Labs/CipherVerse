import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldAlert, Terminal, ArrowLeft, Home, Compass, Lock, KeyRound, Hash, FileCheck, Network, Binary } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function NotFound() {
  const navigate = useNavigate();

  const quickLinks = [
    { name: 'AES Encryption', path: '/symmetric/aes', icon: Lock },
    { name: 'RSA Encryption', path: '/asymmetric/rsa', icon: KeyRound },
    { name: 'Hashing', path: '/hashing', icon: Hash },
    { name: 'Certificates', path: '/certificates', icon: FileCheck },
    { name: 'Blockchain', path: '/blockchain', icon: Network },
    { name: 'API Explorer', path: '/api-explorer', icon: Binary },
  ];

  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center p-6 relative overflow-hidden rounded-xl bg-background border border-border">
      
      {/* Background Cyber Effects */}
      <div className="absolute inset-0 z-0 pointer-events-none opacity-[0.03]" 
        style={{ backgroundImage: 'linear-gradient(#00e5ff 1px, transparent 1px), linear-gradient(90deg, #00e5ff 1px, transparent 1px)', backgroundSize: '40px 40px' }}
      ></div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 blur-[120px] rounded-full pointer-events-none z-0"></div>

      <div className="relative z-10 flex flex-col items-center max-w-2xl text-center w-full">
        {/* Animated Shield Icon */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="mb-8 relative"
        >
          <div className="absolute inset-0 bg-destructive/20 blur-xl rounded-full"></div>
          <ShieldAlert className="w-24 h-24 text-destructive relative z-10" />
        </motion.div>

        {/* 404 Text */}
        <motion.h1 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-7xl font-bold tracking-tighter mb-4 text-foreground font-mono"
        >
          404
        </motion.h1>

        {/* Titles */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h2 className="text-2xl font-semibold mb-2 text-foreground">Page Not Found</h2>
          <p className="text-muted-foreground text-lg mb-2">
            The resource you're looking for does not exist or has been moved.
          </p>
          
          {/* Terminal styled message */}
          <div className="inline-flex items-center gap-2 bg-secondary/50 border border-border px-4 py-2 rounded-md font-mono text-sm text-primary mb-8 shadow-inner">
            <Terminal className="w-4 h-4" />
            <span>Target host could not be resolved.</span>
            <motion.span 
              animate={{ opacity: [1, 0] }} 
              transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
              className="w-2 h-4 bg-primary inline-block ml-1"
            ></motion.span>
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="flex flex-wrap items-center justify-center gap-4 mb-12 w-full"
        >
          <Button onClick={() => navigate('/')} size="lg" className="gap-2 font-semibold bg-primary text-primary-foreground hover:bg-primary/90">
            <Home className="w-4 h-4" />
            Return to Dashboard
          </Button>
          <Button onClick={() => navigate('/symmetric')} variant="outline" size="lg" className="gap-2 bg-background hover:bg-secondary">
            <Compass className="w-4 h-4" />
            Browse Tools
          </Button>
          <Button onClick={() => navigate(-1)} variant="ghost" size="lg" className="gap-2 text-muted-foreground hover:text-foreground">
            <ArrowLeft className="w-4 h-4" />
            Go Back
          </Button>
        </motion.div>

        {/* Smart Suggestions */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="w-full text-left"
        >
          <div className="flex items-center gap-2 mb-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            <div className="h-px bg-border flex-1"></div>
            <span>Quick Access</span>
            <div className="h-px bg-border flex-1"></div>
          </div>
          
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {quickLinks.map((link) => {
              const Icon = link.icon;
              return (
                <Link 
                  key={link.path} 
                  to={link.path}
                  className="flex items-center gap-3 p-3 rounded-lg border border-border bg-card hover:bg-secondary hover:border-primary/50 transition-all duration-200 group"
                >
                  <div className="p-2 rounded-md bg-secondary group-hover:bg-primary/10 group-hover:text-primary transition-colors">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">
                    {link.name}
                  </span>
                </Link>
              );
            })}
          </div>
        </motion.div>

      </div>
    </div>
  );
}
