import { CheckCircle2, XCircle } from 'lucide-react';

export interface ParsedX509 {
  Subject: string;
  Issuer: string;
  'Serial Number': number | string;
  'Not Before': string;
  'Not After': string;
  'Fingerprint SHA256': string;
  Version: string;
  Extensions: string[];
}

export function CertificateReport({ cert }: { cert: ParsedX509 }) {
  const isValid = cert ? new Date(cert['Not Before']) <= new Date() && new Date(cert['Not After']) >= new Date() : false;

  return (
    <div className="space-y-6">
      {/* Status Header Card */}
      <div className="glass rounded-xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-l-4 border-l-primary">
        <div>
          <h3 className="text-lg font-bold text-foreground flex items-center gap-2">
            Certificate Analysis Report
          </h3>
          <p className="text-sm text-muted-foreground mt-1 truncate max-w-md" title={cert.Subject}>
            {cert.Subject}
          </p>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${isValid ? 'bg-success/10 border-success/30 text-success' : 'bg-destructive/10 border-destructive/30 text-destructive'}`}>
          {isValid ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
          <span className="text-sm font-semibold">{isValid ? 'Valid Status' : 'Expired / Invalid'}</span>
        </div>
      </div>

      {/* Data Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass rounded-xl p-5 space-y-4">
          <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">Issuer Details</h4>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">Issued By</span>
            <p className="text-sm font-mono break-words">{cert.Issuer}</p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">Serial Number</span>
            <p className="text-sm font-mono break-all">{cert['Serial Number'].toString()}</p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">Version</span>
            <p className="text-sm font-mono">{cert.Version}</p>
          </div>
        </div>

        <div className="glass rounded-xl p-5 space-y-4">
          <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">Validity Period</h4>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">Not Before (Activation)</span>
            <p className="text-sm font-mono">{new Date(cert['Not Before']).toLocaleString()}</p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">Not After (Expiration)</span>
            <p className={`text-sm font-mono ${isValid ? 'text-success' : 'text-destructive font-bold'}`}>
              {new Date(cert['Not After']).toLocaleString()}
            </p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground font-medium">SHA256 Fingerprint</span>
            <p className="text-xs font-mono break-all text-muted-foreground">{cert['Fingerprint SHA256']}</p>
          </div>
        </div>
      </div>

      {/* Extensions Table */}
      {cert.Extensions && cert.Extensions.length > 0 && (
        <div className="glass rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-background/30">
            <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">X.509 Extensions</h4>
          </div>
          <div className="divide-y divide-border">
            {cert.Extensions.map((ext, idx) => (
              <div key={idx} className="p-4 text-xs font-mono bg-background/50 break-words whitespace-pre-wrap">
                {ext}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Raw JSON View */}
      <div className="glass rounded-xl overflow-hidden mt-6">
        <div className="p-4 border-b border-border bg-background/30">
          <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">Raw JSON Data</h4>
        </div>
        <pre className="p-4 text-[10px] font-mono text-muted-foreground overflow-x-auto bg-[#0a0a0a]">
          {JSON.stringify(cert, null, 2)}
        </pre>
      </div>
    </div>
  );
}
