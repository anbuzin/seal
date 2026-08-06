import { cn } from "@/lib/utils";
import type { Experimental_GeneratedImage } from "ai";

export type ImageProps = Omit<Experimental_GeneratedImage, "uint8Array"> & {
  uint8Array?: Experimental_GeneratedImage["uint8Array"];
  className?: string;
  alt?: string;
};

export const Image = ({ base64, mediaType, className, alt }: ImageProps) => (
  <img
    alt={alt}
    className={cn("h-auto max-w-full overflow-hidden rounded-md", className)}
    src={`data:${mediaType};base64,${base64}`}
  />
);
