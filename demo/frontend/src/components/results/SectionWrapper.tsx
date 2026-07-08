interface SectionWrapperProps {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly children: React.ReactNode;
}

export function SectionWrapper({ id, title, description, children }: SectionWrapperProps) {
  return (
    <section id={id} className="scroll-mt-8 space-y-4">
      <div>
        <h2 className="text-xl font-bold text-white">{title}</h2>
        <p className="mt-1 text-sm text-zinc-400">{description}</p>
      </div>
      {children}
    </section>
  );
}
